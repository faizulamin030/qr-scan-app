"""Web-based DQRC direct payment tool with QR upload support."""

import json
import os
import random
import string
from datetime import datetime
from urllib.parse import parse_qs

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template_string, request


app = Flask(__name__)


BANK_CONFIGS = {
    "MIB": {
        "ip": "10.0.150.36",
        "auth_url": "http://10.0.150.36:9093/authenticate",
        "username": "mcb",
        "password": "mcb",
        "channel": "test",
        "urls": {
            "DQRC": "http://10.0.150.36:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
            "SQRC": "http://10.0.150.36:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC",
        },
    },
    "DIB": {
        "ip": "10.0.150.35",
        "auth_url": "http://10.0.150.35:9093/authenticate",
        "username": "dib",
        "password": "dib",
        "channel": "test",
        "urls": {
            "DQRC": "http://10.0.150.35:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
            "SQRC": "http://10.0.150.35:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC",
        },
    },
    "Telenor": {
        "ip": "10.0.150.38",
        "auth_url": "http://10.0.150.38:9093/authenticate",
        "username": "telenor",
        "password": "telenor",
        "channel": "test",
        "urls": {
            "DQRC": "http://10.0.150.38:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
            "SQRC": "http://10.0.150.38:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC",
        },
    },
}

DEFAULT_BANK = "MIB"


def generate_rrn():
    return "".join(random.choices(string.digits, k=12))


def generate_stan():
    return "".join(random.choices(string.digits, k=6))


def get_datetime():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def parse_qr_data(raw: str) -> dict:
    result = {}
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            result.update(data)
        else:
            result["value"] = data
    except Exception:
        pass

    # Handle key=value;key2=value2 and querystring style content.
    for line in raw.replace(";", "\n").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()

    if "?" in raw and "=" in raw:
        query = raw.split("?", 1)[1]
        for key, values in parse_qs(query).items():
            if values:
                result[key] = values[0]

    # Handle EMV TLV QR format (e.g., 000201010212...)
    emv = parse_emv_tlv(raw)
    if emv:
        result.update(extract_emv_fields(emv))
        result["_emv"] = emv

    # Fallback for variants where normal TLV parsing misses nested 84/02.
    if not result.get("rtpId"):
        rtp_fallback = extract_rtp_from_raw(raw)
        if rtp_fallback:
            result["rtpId"] = rtp_fallback

    result["_raw"] = raw
    return result


def extract_rtp_from_raw(raw: str) -> str:
    # Scan the complete raw string for any 84<LL><payload> and then 02<LL><value> inside payload.
    i = 0
    while i + 4 <= len(raw):
        if raw[i:i + 2] == "84" and raw[i + 2:i + 4].isdigit():
            payload_len = int(raw[i + 2:i + 4])
            start = i + 4
            end = start + payload_len
            if end <= len(raw):
                payload = raw[start:end]
                rtp = extract_subtag_value(payload, "02")
                if rtp:
                    return rtp
                # Some payloads nest additional TLV levels.
                nested = parse_emv_tlv(payload)
                if nested:
                    rtp_nested = find_emv_subtag_under_tag(nested, parent_tag="84", child_tag="02")
                    if rtp_nested:
                        return rtp_nested
            i += 2
            continue
        i += 1
    return ""


def extract_subtag_value(tlv_payload: str, sub_tag: str) -> str:
    j = 0
    while j + 4 <= len(tlv_payload):
        tag = tlv_payload[j:j + 2]
        ln = tlv_payload[j + 2:j + 4]
        if not (tag.isdigit() and ln.isdigit()):
            j += 1
            continue
        value_len = int(ln)
        value_start = j + 4
        value_end = value_start + value_len
        if value_end > len(tlv_payload):
            # Keep scanning; there may be another valid TLV frame later in payload.
            j += 1
            continue
        value = tlv_payload[value_start:value_end]
        if tag == sub_tag:
            return value
        j = value_end
    return ""


def parse_emv_tlv(raw: str) -> dict:
    if not raw or len(raw) < 8:
        return {}

    out = {}
    idx = 0
    while idx + 4 <= len(raw):
        tag = raw[idx:idx + 2]
        ln = raw[idx + 2:idx + 4]
        if not (tag.isdigit() and ln.isdigit()):
            # Keep previously parsed tags; some scanners return noisy tail bytes.
            break

        value_len = int(ln)
        idx += 4
        if idx + value_len > len(raw):
            # Incomplete payload can happen in copied samples; keep parsed data.
            break

        value = raw[idx:idx + value_len]
        out[tag] = value
        idx += value_len

    return out


def extract_emv_fields(emv: dict) -> dict:
    fields = {}

    # Standard top-level EMV tags commonly used by merchant-presented QR.
    if emv.get("52"):
        fields["categoryCode"] = emv["52"]
        fields["mcc"] = emv["52"]
    if emv.get("54"):
        fields["amount"] = emv["54"]
    if emv.get("59"):
        fields["accountTitle"] = emv["59"]
        fields["merchantTitle"] = emv["59"]
        fields["dba"] = emv["59"]
    if emv.get("60"):
        fields["city"] = emv["60"]

    # Merchant account information templates are tags 26-51.
    for tag_num in range(26, 52):
        tag = f"{tag_num:02d}"
        if tag not in emv:
            continue
        nested = parse_emv_tlv(emv[tag])
        if not nested:
            continue

        # In many local QR specs, nested 01=BIC and 02=IBAN.
        if nested.get("01"):
            fields.setdefault("bic", nested["01"])
        if nested.get("02"):
            fields.setdefault("iban", nested["02"])
            fields.setdefault("merchantIBAN", nested["02"])
        if nested.get("00"):
            fields.setdefault("merchantIdAlias", nested["00"])

    # Additional data field template (62) has store/terminal labels in sub-tags.
    if emv.get("62"):
        add = parse_emv_tlv(emv["62"])
        if add.get("03"):
            fields["storeLabel"] = add["03"]
        if add.get("07"):
            fields["terminalID"] = add["07"]
        if add.get("05"):
            fields["referenceLabel"] = add["05"]

    # RTP ID from custom tag 84, sub-tag 01 (actual location) or 02 (fallback).
    rtp_from_84 = ""
    if emv.get("84"):
        # 1) Direct parse when tag84 payload itself is TLV.
        tag_84 = parse_emv_tlv(emv["84"])
        # Check sub-tag 01 first (actual RTP location), then 02 as fallback.
        if tag_84.get("01"):
            rtp_from_84 = tag_84["01"]
        elif tag_84.get("02"):
            rtp_from_84 = tag_84["02"]

        # 2) Tolerant scan within raw tag84 payload in case of non-TLV prefixes.
        if not rtp_from_84:
            rtp_from_84 = extract_subtag_value(emv["84"], "01")
            if not rtp_from_84:
                rtp_from_84 = extract_subtag_value(emv["84"], "02")

    # 3) Recursive search when tag84 is nested under another template.
    if not rtp_from_84:
        rtp_from_84 = find_emv_subtag_under_tag(emv, parent_tag="84", child_tag="01")
        if not rtp_from_84:
            rtp_from_84 = find_emv_subtag_under_tag(emv, parent_tag="84", child_tag="02")

    if rtp_from_84:
        fields["rtpId"] = rtp_from_84

    return fields


def find_emv_subtag_under_tag(tlv: dict, parent_tag: str, child_tag: str, depth: int = 0) -> str:
    if depth > 12:
        return ""

    for tag, value in tlv.items():
        if tag == parent_tag:
            nested = parse_emv_tlv(value)
            if nested.get(child_tag):
                return nested[child_tag]

        nested_any = parse_emv_tlv(value)
        if nested_any:
            found = find_emv_subtag_under_tag(nested_any, parent_tag, child_tag, depth + 1)
            if found:
                return found

    return ""


def flatten_dict(data: dict, parent_key: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        full_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
            flat[str(key)] = json.dumps(value)
        else:
            flat[full_key] = value
            flat[str(key)] = value
    return flat


def map_qr_to_form(parsed: dict, data: dict) -> list:
    # Normalize keys so mapping works with nested JSON and key casing variations.
    flat = flatten_dict(parsed)
    flat_lower = {str(k).lower(): "" if v is None else str(v) for k, v in flat.items()}

    candidates = {
        "merch_iban": ["merchantiban", "iban", "merchantinfo.iban"],
        "merch_title": ["merchanttitle", "accounttitle", "merchantinfo.accounttitle"],
        "merch_dba": ["dba", "merchantinfo.dba"],
        "merch_bic": ["bic", "merchantinfo.bic"],
        "merch_mcc": ["mcc", "categorycode", "merchantinfo.categorycode"],
        "merch_alias": ["merchantidalias", "merchantinfo.merchantidalias"],
        "merch_tax": ["merchanttaxid", "additionalinfo.merchanttaxid"],
        "merch_chan": ["merchantchannel", "additionalinfo.merchantchannel"],
        "store_label": ["storelabel", "additionalinfo.storelabel"],
        "terminal_id": ["terminalid", "additionalinfo.terminalid"],
        "rtp_id": ["rtpid", "additionalinfo.rtpid"],
        "city": ["city", "additionalinfo.city"],
        "amount": ["amount", "amountinfo.amount"],
    }

    filled = []
    for target, keys in candidates.items():
        for key in keys:
            value = flat_lower.get(key)
            if value:
                data[target] = value
                filled.append(f"{target}<= {key}")
                break

    return filled


QR_AUTOFILL_FIELDS = [
    "merch_iban",
    "merch_title",
    "merch_dba",
    "merch_bic",
    "merch_mcc",
    "merch_alias",
    "merch_tax",
    "merch_chan",
    "store_label",
    "terminal_id",
    "rtp_id",
    "city",
    "amount",
    "ref_label",
    "qr_raw",
]


def clear_autofilled_fields(data: dict) -> None:
    for key in QR_AUTOFILL_FIELDS:
        data[key] = ""


def extract_auth_token(response_body) -> str:
    if isinstance(response_body, str):
        token = response_body.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        return token

    if isinstance(response_body, dict):
        for key in ["token", "accessToken", "access_token", "bearerToken", "jwt", "id_token"]:
            value = response_body.get(key)
            if value:
                token = str(value).strip()
                if token.lower().startswith("bearer "):
                    token = token[7:].strip()
                return token

        for nested_key in ["data", "result", "response"]:
            nested = response_body.get(nested_key)
            if isinstance(nested, dict):
                nested_token = extract_auth_token(nested)
                if nested_token:
                    return nested_token

    return ""




FIELDS = {
    "bank": DEFAULT_BANK,
    "txn_type": "DQRC",
    "url": BANK_CONFIGS[DEFAULT_BANK]["urls"]["DQRC"],
    "auth_url": BANK_CONFIGS[DEFAULT_BANK]["auth_url"],
    "token": "",
    "auth_username": "mcb",
    "auth_password": "mcb",
    "auth_channel": "test",
    "rrn": "",
    "stan": "",
    "sender_id_type": "",
    "sender_id_value": "",
    "sender_iban": "",
    "sender_title": "",
    "cust_mobile": "",
    "cust_email": "",
    "cust_address": "",
    "cust_label": "",
    "amount": "0",
    "tip": "0",
    "fee": "0",
    "latitude": "",
    "longitude": "",
    "merch_title": "",
    "merch_dba": "",
    "merch_bic": "",
    "merch_iban": "",
    "merch_mcc": "",
    "merch_alias": "",
    "merch_tax": "",
    "merch_chan": "",
    "store_label": "",
    "terminal_id": "",
    "rtp_id": "",
    "city": "Karachi",
    "country": "PK",
    "currency": "586",
    "bill_number": "",
    "bill_due": "",
    "loyalty": "",
    "ref_label": "",
    "ttc": "",
    "purpose": "",
    "amt_after": "0",
    "qr_raw": "",
}


QR_MAPPING = {
    "merchantIBAN": "merch_iban",
    "iban": "merch_iban",
    "merchantTitle": "merch_title",
    "accountTitle": "merch_title",
    "dba": "merch_dba",
    "bic": "merch_bic",
    "mcc": "merch_mcc",
    "categoryCode": "merch_mcc",
    "merchantIdAlias": "merch_alias",
    "merchantTaxId": "merch_tax",
    "merchantChannel": "merch_chan",
    "storeLabel": "store_label",
    "terminalID": "terminal_id",
    "rtpId": "rtp_id",
    "city": "city",
    "amount": "amount",
}


def float_value(value: str) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def form_data() -> dict:
    data = FIELDS.copy()
    for key in data:
        if key in request.form:
            data[key] = request.form.get(key, "").strip()
    
    bank = (data.get("bank") or DEFAULT_BANK).strip()
    if bank not in BANK_CONFIGS:
        bank = DEFAULT_BANK
    data["bank"] = bank
    
    txn_type = (data.get("txn_type") or "DQRC").upper()
    if txn_type not in ["DQRC", "SQRC"]:
        txn_type = "DQRC"
    data["txn_type"] = txn_type
    
    # Endpoint and credentials follow selected bank and transaction type.
    if not data["url"]:
        data["url"] = BANK_CONFIGS[bank]["urls"][txn_type]
    if not data["auth_url"]:
        data["auth_url"] = BANK_CONFIGS[bank]["auth_url"]
    if not data["auth_username"]:
        data["auth_username"] = BANK_CONFIGS[bank]["username"]
    if not data["auth_password"]:
        data["auth_password"] = BANK_CONFIGS[bank]["password"]
    if not data["auth_channel"]:
        data["auth_channel"] = BANK_CONFIGS[bank]["channel"]
    
    if not data["rrn"]:
        data["rrn"] = generate_rrn()
    if not data["stan"]:
        data["stan"] = generate_stan()
    return data


def payload(data: dict) -> dict:
    amt = float_value(data["amount"])
    tip = float_value(data["tip"])
    fee = float_value(data["fee"])
    total = str(int(amt + tip + fee))
    return {
        "info": {"rrn": data["rrn"], "stan": data["stan"]},
        "senderIdentification": {
            "idType": data["sender_id_type"],
            "idValue": data["sender_id_value"],
        },
        "senderInfo": {
            "iban": data["sender_iban"],
            "accountTitle": data["sender_title"],
        },
        "paymentInfo": {
            "longitude": data["longitude"],
            "latitude": data["latitude"],
            "dateTime": get_datetime(),
        },
        "amountInfo": {
            "amount": amt,
            "tip": tip,
            "totalAmount": total,
            "fee": fee,
        },
        "merchantInfo": {
            "accountTitle": data["merch_title"],
            "dba": data["merch_dba"],
            "categoryCode": data["merch_mcc"],
            "bic": data["merch_bic"],
            "iban": data["merch_iban"],
            "merchantIdAlias": data["merch_alias"],
        },
        "additionalInfo": {
            "currency": data["currency"],
            "country": data["country"],
            "city": data["city"],
            "billNumber": data["bill_number"],
            "billDueDate": data["bill_due"],
            "loyaltyNumber": data["loyalty"],
            "mobileNumber": data["cust_mobile"],
            "storeLabel": data["store_label"],
            "customerLabel": data["cust_label"],
            "terminalID": data["terminal_id"],
            "paymentPurpose": data["purpose"],
            "merchantTaxId": data["merch_tax"],
            "merchantChannel": data["merch_chan"],
            "rtpId": None if data.get("txn_type") == "SQRC" else data["rtp_id"],
            "ttc": data["ttc"],
            "customerMobile": data["cust_mobile"],
            "customerEmail": data["cust_email"],
            "customerAddress": data["cust_address"],
            "referenceLabel": data["ref_label"],
            "amountAfterDueDate": float_value(data["amt_after"]),
        },
        "reserveFields": {"r1": "", "r2": "", "r3": "", "r4": "", "r5": ""},
    }


def scan_qr(uploaded_file) -> str:
    image_bytes = uploaded_file.read()
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        return ""

    detector = cv2.QRCodeDetector()

    # Try default decode first.
    decoded_text, _, _ = detector.detectAndDecode(image)
    if decoded_text:
      return decoded_text.strip()

    # Fallback: grayscale and high-contrast threshold can recover hard-to-read QR images.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    decoded_text, _, _ = detector.detectAndDecode(binary)
    if decoded_text:
      return decoded_text.strip()

    # Last fallback: detect multiple QR codes and use the first decoded item.
    ok, decoded_list, _, _ = detector.detectAndDecodeMulti(image)
    if ok and decoded_list:
      for item in decoded_list:
        if item:
          return item.strip()

    return ""


@app.route("/auth-token", methods=["POST"])
def auth_token():
    req = request.get_json(silent=True) or {}
    username = str(req.get("username", "")).strip()
    password = str(req.get("password", "")).strip()
    channel = str(req.get("channel", "")).strip()
    auth_url_custom = str(req.get("auth_url", "")).strip()

    if not username or not password or not channel:
        return jsonify({"ok": False, "message": "Username, password and channel are required."}), 400

    auth_payload = {
        "username": username,
        "password": password,
        "channel": channel,
    }
    
    # Use custom URL if provided, else fall back to default (MIB).
    url_to_use = auth_url_custom if auth_url_custom else BANK_CONFIGS[DEFAULT_BANK]["auth_url"]

    try:
        response = requests.post(
            url_to_use,
            headers={
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            json=auth_payload,
            timeout=30,
        )
        try:
            auth_response = response.json()
        except Exception:
            auth_response = response.text
        
        token = extract_auth_token(auth_response)
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 502

    if not token:
        return jsonify(
            {
                "ok": False,
                "statusCode": response.status_code,
                "message": "Authenticate API did not return a token.",
                "response": auth_response,
            }
        ), 502

    return jsonify(
        {
            "ok": True,
            "statusCode": response.status_code,
            "token": token,
            "response": auth_response,
        }
    )


@app.route("/", methods=["GET", "POST"])
def index():
    data = form_data() if request.method == "POST" else FIELDS.copy()
    if not data["rrn"]:
        data["rrn"] = generate_rrn()
    if not data["stan"]:
        data["stan"] = generate_stan()

    status = "IDLE"
    log_lines = []
    response_text = ""

    action = request.form.get("action", "") if request.method == "POST" else ""

    if action == "newids":
        data["rrn"] = generate_rrn()
        data["stan"] = generate_stan()
        status = "NEW IDS GENERATED"
        log_lines.append("Generated fresh RRN and STAN.")

    if action == "scan":
        qr_file = request.files.get("qr_file")
        if not qr_file or not qr_file.filename:
            status = "ERROR: No file selected"
            log_lines.append("Please upload an image file first.")
        else:
            clear_autofilled_fields(data)
            raw = scan_qr(qr_file)
            if raw:
                status = "QR DETECTED"
                data["qr_raw"] = raw[:80]
                parsed = parse_qr_data(raw)
                filled = map_qr_to_form(parsed, data)
                if data.get("txn_type") == "SQRC":
                    data["rtp_id"] = ""
                log_lines.append(f"QR scanned: {raw[:120]}")
                if parsed.get("_emv"):
                    emv_keys = ", ".join(sorted(parsed["_emv"].keys()))
                    log_lines.append(f"EMV tags: {emv_keys}")
                    if "84" in parsed["_emv"]:
                        log_lines.append(f"EMV[84]: {parsed['_emv']['84']}")
                        emv84_nested = parse_emv_tlv(parsed["_emv"]["84"])
                        if emv84_nested:
                            nested_keys = ", ".join(sorted(emv84_nested.keys()))
                            log_lines.append(f"EMV[84] subtags: {nested_keys}")
                if parsed.get("rtpId"):
                    if data.get("txn_type") == "SQRC":
                        log_lines.append("RTP found in QR but ignored for SQRC.")
                    else:
                        log_lines.append(f"RTP from QR: {parsed['rtpId']}")
                elif parsed.get("_emv") and "84" in parsed["_emv"]:
                    fallback_rtp = extract_subtag_value(parsed["_emv"]["84"], "02")
                    log_lines.append(f"RTP(84/02) fallback: {fallback_rtp or 'NOT FOUND'}")
                if filled:
                    log_lines.append(f"Auto-filled: {', '.join(filled)}")
                else:
                  sample_keys = ", ".join(sorted([k for k in parsed.keys() if not k.startswith("_")])[:12])
                  log_lines.append("QR parsed; no recognized fields found.")
                  if sample_keys:
                    log_lines.append(f"Parsed keys: {sample_keys}")
            else:
                status = "NO QR FOUND"
                log_lines.append("No QR code detected in uploaded image.")

    if action == "execute":
        if not data["token"]:
            status = "ERROR"
            log_lines.append("Missing Bearer token.")
        elif not data["url"]:
            status = "ERROR"
            log_lines.append("Missing API URL.")
        else:
            req_payload = payload(data)
            log_lines.append(f"POST {data['url']}")
            log_lines.append(json.dumps(req_payload, indent=2))
            try:
                r = requests.post(
                    data["url"],
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "*/*",
                        "Authorization": f"Bearer {data['token']}",
                    },
                    json=req_payload,
                    timeout=30,
                )
                try:
                    response_text = json.dumps(r.json(), indent=2)
                except Exception:
                    response_text = r.text
                status = f"HTTP {r.status_code}"
                data["rrn"] = generate_rrn()
                data["stan"] = generate_stan()
            except Exception as exc:
                status = "ERROR"
                response_text = str(exc)

    return render_template_string(
        TEMPLATE,
        data=data,
        status=status,
        logs="\n".join(log_lines),
        response=response_text,
        auth_url=data.get("auth_url", BANK_CONFIGS[DEFAULT_BANK]["auth_url"]),
    )


TEMPLATE = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>DQRC Web Terminal</title>
  <style>
        :root {
            --bg: #0b111a;
            --bg-glow: #18283d;
            --panel: #111a27;
            --panel-2: #0f1724;
            --line: #2a3a52;
            --line-strong: #3b526f;
            --txt: #ecf3fb;
            --muted: #b5c2d4;
            --accent: #7dd3fc;
            --accent-soft: rgba(125, 211, 252, 0.22);
            --ok: #34d399;
            --warn: #f59e0b;
            --shadow: 0 12px 28px rgba(0, 0, 0, 0.30);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background:
                radial-gradient(1200px 700px at 92% -10%, var(--bg-glow) 0%, transparent 55%),
                linear-gradient(180deg, #0b121d 0%, var(--bg) 100%);
            color: var(--txt);
            font-family: "Segoe UI", "Inter", sans-serif;
            font-size: 14px;
            line-height: 1.45;
            -webkit-font-smoothing: antialiased;
        }
        .wrap {
            max-width: 1220px;
            margin: 0 auto;
            padding: 24px;
        }
        h1 {
            margin: 0 0 6px 0;
            color: var(--txt);
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .sub {
            color: var(--muted);
            margin-bottom: 20px;
            font-size: 14px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 14px;
            align-items: start;
        }
        .panel {
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px;
            box-shadow: var(--shadow);
            min-width: 0;
            overflow: hidden;
        }
        .panel h3 {
            margin: 0 0 12px 0;
            color: var(--accent);
            font-size: 15px;
            font-weight: 650;
            letter-spacing: 0.2px;
        }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 10px;
            align-items: start;
            min-width: 0;
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 0;
        }
        .row > .field { min-width: 0; }
        label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.2px;
            display: block;
            line-height: 1.35;
            word-break: break-word;
        }
        input,
        select,
        textarea {
            width: 100%;
            max-width: 100%;
            background: #0d1522;
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--txt);
            padding: 10px 11px;
            font-size: 14px;
            line-height: 1.3;
            transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
        }
        input:hover,
        select:hover,
        textarea:hover {
            border-color: var(--line-strong);
        }
        input:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-soft);
            background: #0e1827;
        }
        .full { grid-column: 1 / -1; }
        .actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 12px 0;
            min-width: 0;
        }
        button {
            border: 1px solid var(--accent);
            background: rgba(125, 211, 252, 0.08);
            color: var(--accent);
            border-radius: 8px;
            padding: 9px 14px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: transform 0.15s ease, background-color 0.2s ease, border-color 0.2s ease;
        }
        button:hover {
            transform: translateY(-1px);
            background: rgba(125, 211, 252, 0.16);
        }
        button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 3px var(--accent-soft);
        }
        button.auth {
            border-color: var(--warn);
            color: var(--warn);
            background: rgba(245, 158, 11, 0.08);
        }
        button.auth:hover {
            background: rgba(245, 158, 11, 0.16);
        }
        button.exec {
            border-color: var(--ok);
            color: var(--ok);
            background: rgba(52, 211, 153, 0.08);
        }
        button.exec:hover {
            background: rgba(52, 211, 153, 0.16);
        }
        .status {
            margin-top: 8px;
            font-weight: 600;
            color: var(--txt);
        }
        textarea {
            width: 100%;
            max-width: 100%;
            min-height: 140px;
            resize: vertical;
            line-height: 1.4;
        }
        .modal {
            position: fixed;
            inset: 0;
            background: rgba(5, 10, 18, 0.72);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 16px;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s ease, visibility 0.2s ease;
        }
        .modal.open {
            opacity: 1;
            visibility: visible;
        }
        .modal-card {
            width: min(560px, 100%);
            background: linear-gradient(180deg, #132033 0%, #101a2a 100%);
            border: 1px solid var(--line-strong);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
            transform: translateY(8px) scale(0.98);
            transition: transform 0.22s ease;
        }
        .modal.open .modal-card {
            transform: translateY(0) scale(1);
        }
        .modal-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            gap: 12px;
        }
        .modal-title {
            color: var(--txt);
            font-weight: 700;
            font-size: 16px;
        }
        .auth-note {
            color: var(--muted);
            font-size: 12px;
            margin-top: 4px;
            min-height: 16px;
        }
        .close-btn {
            border-color: var(--line-strong);
            color: var(--txt);
            background: rgba(236, 243, 251, 0.04);
        }
        @media (max-width: 900px) {
            .wrap { padding: 16px; }
            .grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 640px) {
            h1 { font-size: 24px; }
            .row { grid-template-columns: 1fr; gap: 8px; }
            .panel { padding: 12px; border-radius: 10px; }
            .field { gap: 5px; }
            button { width: 100%; }
            .actions { gap: 8px; }
        }
  </style>
</head>
<body>
  <div class=\"wrap\">
        <h1>QR Web Terminal</h1>
        <div class=\"sub\">Select DQRC or SQRC, upload QR image, review fields, then execute payment</div>
    <form method=\"post\" enctype=\"multipart/form-data\">
      <div class=\"panel\">
        <h3>Endpoint</h3>
        <div class=\"row\">
          <div class=\"field\">
            <label>Bank</label>
            <select name=\"bank\">
              <option value=\"MIB\" {% if data.bank == 'MIB' %}selected{% endif %}>MIB (10.0.150.36)</option>
              <option value=\"DIB\" {% if data.bank == 'DIB' %}selected{% endif %}>DIB (10.0.150.35)</option>
              <option value=\"Telenor\" {% if data.bank == 'Telenor' %}selected{% endif %}>Telenor (10.0.150.38)</option>
            </select>
          </div>
          <div class=\"field\">
            <label>Transaction Type</label>
            <select name=\"txn_type\">
              <option value=\"DQRC\" {% if data.txn_type == 'DQRC' %}selected{% endif %}>DQRC</option>
              <option value=\"SQRC\" {% if data.txn_type == 'SQRC' %}selected{% endif %}>SQRC</option>
            </select>
          </div>
          <div class=\"field full\"><label>API URL</label><input name=\"url\" value=\"{{data.url}}\"></div>
          <div class=\"field full\"><label>Bearer Token</label><input id=\"tokenInput\" name=\"token\" value=\"{{data.token}}\"></div>
        </div>
        <div class=\"actions\" style=\"margin-top:0\">
          <button class=\"auth\" type=\"button\" id=\"openAuthModal\">Generate Bearer Token</button>
        </div>
      </div>

      <div class=\"panel\" style=\"margin-top:12px\">
        <h3>QR Upload</h3>
        <div class=\"row\">
          <div class=\"field full\"><label>Upload Image</label><input type=\"file\" name=\"qr_file\" accept=\".jpg,.jpeg,.png,.bmp,.gif\"></div>
          <div class=\"field full\"><label>QR Raw</label><input name=\"qr_raw\" value=\"{{data.qr_raw}}\" readonly></div>
        </div>
        <div class=\"actions\">
          <button type=\"submit\" name=\"action\" value=\"scan\">Scan Uploaded QR</button>
          <button type=\"submit\" name=\"action\" value=\"newids\">New IDs</button>
          <button class=\"exec\" type=\"submit\" name=\"action\" value=\"execute\">Execute Payment</button>
        </div>
        <div class=\"status\">Status: {{status}}</div>
      </div>

      <div class=\"grid\" style=\"margin-top:12px\">
        <div class=\"panel\">
          <h3>Transaction + Sender</h3>
          <div class=\"row\">
            <div class=\"field\"><label>RRN</label><input name=\"rrn\" value=\"{{data.rrn}}\"></div>
            <div class=\"field\"><label>STAN</label><input name=\"stan\" value=\"{{data.stan}}\"></div>
            <div class=\"field\"><label>ID Type</label><input name=\"sender_id_type\" value=\"{{data.sender_id_type}}\"></div>
            <div class=\"field\"><label>ID Value</label><input name=\"sender_id_value\" value=\"{{data.sender_id_value}}\"></div>
            <div class=\"field\"><label>Sender IBAN</label><input name=\"sender_iban\" value=\"{{data.sender_iban}}\"></div>
            <div class=\"field\"><label>Sender Title</label><input name=\"sender_title\" value=\"{{data.sender_title}}\"></div>
            <div class=\"field\"><label>Mobile</label><input name=\"cust_mobile\" value=\"{{data.cust_mobile}}\"></div>
            <div class=\"field\"><label>Email</label><input name=\"cust_email\" value=\"{{data.cust_email}}\"></div>
            <div class=\"field full\"><label>Address</label><input name=\"cust_address\" value=\"{{data.cust_address}}\"></div>
            <div class=\"field full\"><label>Customer Label</label><input name=\"cust_label\" value=\"{{data.cust_label}}\"></div>
          </div>
        </div>

        <div class=\"panel\">
          <h3>Amounts + Location</h3>
          <div class=\"row\">
            <div class=\"field\"><label>Amount</label><input name=\"amount\" value=\"{{data.amount}}\"></div>
            <div class=\"field\"><label>Tip</label><input name=\"tip\" value=\"{{data.tip}}\"></div>
            <div class=\"field\"><label>Fee</label><input name=\"fee\" value=\"{{data.fee}}\"></div>
            <div class=\"field\"><label>Amount After Due</label><input name=\"amt_after\" value=\"{{data.amt_after}}\"></div>
            <div class=\"field\"><label>Latitude</label><input name=\"latitude\" value=\"{{data.latitude}}\"></div>
            <div class=\"field\"><label>Longitude</label><input name=\"longitude\" value=\"{{data.longitude}}\"></div>
            <div class=\"field\"><label>Currency</label><input name=\"currency\" value=\"{{data.currency}}\"></div>
            <div class=\"field\"><label>Country</label><input name=\"country\" value=\"{{data.country}}\"></div>
            <div class=\"field\"><label>City</label><input name=\"city\" value=\"{{data.city}}\"></div>
            <div class=\"field\"><label>Purpose</label><input name=\"purpose\" value=\"{{data.purpose}}\"></div>
            <div class=\"field\"><label>Bill Number</label><input name=\"bill_number\" value=\"{{data.bill_number}}\"></div>
            <div class=\"field\"><label>Bill Due</label><input name=\"bill_due\" value=\"{{data.bill_due}}\"></div>
            <div class=\"field\"><label>Loyalty</label><input name=\"loyalty\" value=\"{{data.loyalty}}\"></div>
            <div class=\"field\"><label>Reference Label</label><input name=\"ref_label\" value=\"{{data.ref_label}}\"></div>
            <div class=\"field full\"><label>TTC</label><input name=\"ttc\" value=\"{{data.ttc}}\"></div>
          </div>
        </div>

        <div class=\"panel\">
          <h3>Merchant + Routing</h3>
          <div class=\"row\">
            <div class=\"field\"><label>Merchant Title</label><input name=\"merch_title\" value=\"{{data.merch_title}}\"></div>
            <div class=\"field\"><label>DBA</label><input name=\"merch_dba\" value=\"{{data.merch_dba}}\"></div>
            <div class=\"field\"><label>BIC</label><input name=\"merch_bic\" value=\"{{data.merch_bic}}\"></div>
            <div class=\"field\"><label>Merchant IBAN</label><input name=\"merch_iban\" value=\"{{data.merch_iban}}\"></div>
            <div class=\"field\"><label>MCC</label><input name=\"merch_mcc\" value=\"{{data.merch_mcc}}\"></div>
            <div class=\"field\"><label>Alias</label><input name=\"merch_alias\" value=\"{{data.merch_alias}}\"></div>
            <div class=\"field\"><label>Tax ID</label><input name=\"merch_tax\" value=\"{{data.merch_tax}}\"></div>
            <div class=\"field\"><label>Channel</label><input name=\"merch_chan\" value=\"{{data.merch_chan}}\"></div>
            <div class=\"field\"><label>Store Label</label><input name=\"store_label\" value=\"{{data.store_label}}\"></div>
            <div class=\"field\"><label>Terminal ID</label><input name=\"terminal_id\" value=\"{{data.terminal_id}}\"></div>
            <div class=\"field full\"><label>RTP ID</label><input name=\"rtp_id\" value=\"{{data.rtp_id}}\"></div>
          </div>
        </div>
      </div>

      <div class=\"grid\" style=\"margin-top:12px\">
        <div class=\"panel\"><h3>Log</h3><textarea readonly>{{logs}}</textarea></div>
        <div class=\"panel\"><h3>Response</h3><textarea readonly>{{response}}</textarea></div>
      </div>

            <div class=\"modal\" id=\"authModal\">
                <div class=\"modal-card\">
                    <div class=\"modal-head\">
                        <div class=\"modal-title\">Authenticate and Generate Token</div>
                        <button class=\"close-btn\" type=\"button\" id=\"closeAuthModal\">Close</button>
                    </div>
                    <div class=\"row\">
                        <div class="field full"><label>Authenticate URL</label><input id="authUrl" value="{{auth_url}}"></div>
                        <div class=\"field\"><label>Username</label><input id=\"authUsername\" name=\"auth_username\" value=\"{{data.auth_username}}\"></div>
                        <div class=\"field\"><label>Password</label><input id=\"authPassword\" name=\"auth_password\" value=\"{{data.auth_password}}\"></div>
                        <div class=\"field full\"><label>Channel</label><input id=\"authChannel\" name=\"auth_channel\" value=\"{{data.auth_channel}}\"></div>
                    </div>
                    <div class=\"actions\">
                        <button class=\"auth\" type=\"button\" id=\"generateTokenBtn\">Generate Token</button>
                    </div>
                    <div class=\"auth-note\" id=\"authStatus\"></div>
                </div>
            </div>
    </form>
  </div>

    <script>
        // Bank and transaction type configurations with credentials
        const bankConfigs = {
            "MIB": {
                "auth_url": "http://10.0.150.36:9093/authenticate",
                "username": "mcb",
                "password": "mcb",
                "channel": "test",
                "urls": {
                    "DQRC": "http://10.0.150.36:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
                    "SQRC": "http://10.0.150.36:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC"
                }
            },
            "DIB": {
                "auth_url": "http://10.0.150.35:9093/authenticate",
                "username": "dib",
                "password": "dib",
                "channel": "test",
                "urls": {
                    "DQRC": "http://10.0.150.35:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
                    "SQRC": "http://10.0.150.35:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC"
                }
            },
            "Telenor": {
                "auth_url": "http://10.0.150.38:9093/authenticate",
                "username": "telenor",
                "password": "telenor",
                "channel": "test",
                "urls": {
                    "DQRC": "http://10.0.150.38:9093/api/v1/paysyslabs/merchant/payment/direct/DQRC",
                    "SQRC": "http://10.0.150.38:9093/api/v1/paysyslabs/merchant/payment/direct/SQRC"
                }
            }
        };

        const authModal = document.getElementById("authModal");
        const openAuthModal = document.getElementById("openAuthModal");
        const closeAuthModal = document.getElementById("closeAuthModal");
        const generateTokenBtn = document.getElementById("generateTokenBtn");
        const authStatus = document.getElementById("authStatus");
        const tokenInput = document.getElementById("tokenInput");
        const bankSelect = document.querySelector("select[name='bank']");
        const txnTypeSelect = document.querySelector("select[name='txn_type']");
        const urlInput = document.querySelector("input[name='url']");
        const authUrlInput = document.getElementById("authUrl");
        const authUsernameInput = document.getElementById("authUsername");
        const authPasswordInput = document.getElementById("authPassword");
        const authChannelInput = document.getElementById("authChannel");

        // Function to update endpoints AND credentials based on bank and txn_type
        function updateEndpoints() {
            const bank = bankSelect.value;
            const txnType = txnTypeSelect.value;
            
            if (bankConfigs[bank]) {
                urlInput.value = bankConfigs[bank].urls[txnType] || "";
                authUrlInput.value = bankConfigs[bank].auth_url || "";
                authUsernameInput.value = bankConfigs[bank].username || "";
                authPasswordInput.value = bankConfigs[bank].password || "";
                authChannelInput.value = bankConfigs[bank].channel || "";
            }
        }

        // Listen for bank and transaction type changes
        if (bankSelect) {
            bankSelect.addEventListener("change", updateEndpoints);
        }
        if (txnTypeSelect) {
            txnTypeSelect.addEventListener("change", updateEndpoints);
        }

        openAuthModal.addEventListener("click", function () {
            authStatus.textContent = "";
            authModal.classList.add("open");
        });

        closeAuthModal.addEventListener("click", function () {
            authModal.classList.remove("open");
        });

        authModal.addEventListener("click", function (event) {
            if (event.target === authModal) {
                authModal.classList.remove("open");
            }
        });

        generateTokenBtn.addEventListener("click", async function () {
            const auth_url = document.getElementById("authUrl").value.trim();
            const username = document.getElementById("authUsername").value.trim();
            const password = document.getElementById("authPassword").value.trim();
            const channel = document.getElementById("authChannel").value.trim();

            if (!auth_url || !username || !password || !channel) {
                authStatus.textContent = "Authenticate URL, username, password and channel are required.";
                return;
            }

            generateTokenBtn.disabled = true;
            authStatus.textContent = "Generating token...";

            try {
                const response = await fetch("/auth-token", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    body: JSON.stringify({ auth_url, username, password, channel })
                });

                const result = await response.json();
                if (!response.ok || !result.ok) {
                    authStatus.textContent = result.message || "Failed to generate token.";
                    return;
                }

                tokenInput.value = result.token || "";
                authStatus.textContent = "Token generated and filled in Bearer Token field.";
                authModal.classList.remove("open");
            } catch (error) {
                authStatus.textContent = "Authentication request failed.";
            } finally {
                generateTokenBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host=host, port=port)