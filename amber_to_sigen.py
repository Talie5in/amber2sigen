#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amber (5-minute or 30-minute) -> Sigen staticPricing (now -> +24h)

Version: v29
- Added SELL channel selection via --sell-channel (general|feedIn).
- Implemented SELL advanced price support for feed-in via --sell-advanced-price / SELLADVANCED.
  - Supports advancedPrice.(low|predicted|high) for feed-in.
  - Automatically flips sign (Amber feed-in advanced prices are negative).
  - Falls back to feed-in spotPerKwh if advanced value missing.
- Improved /prices/current override logic:
  - BUY always sourced from channelType=general.
  - SELL sourced from selected channelType (general or feedIn).
  - Independent BUY/SELL row selection with robust label matching.
- Added per-channel debug summaries for current window:
  - BUY: shows channel + advanced level (or off).
  - SELL: shows channel + advanced level (or off).
- Safer SELL override handling:
  - Skips override when SELL value missing instead of forcing zero.
  - Explicit logging when fallback logic is used.
- Minor logging, comments, and guardrail cleanups.

Version: v28
- Refactored BUY/SELL separation throughout pipeline.
  - BUY strictly uses Amber "general" channel.
  - SELL channel made configurable but defaults to "general" for backward compatibility.
- Added SELL baseline carry-forward logic (mirrors BUY behavior).
- Hardened current-triplet selection to better handle mixed channelType responses.
- Improved debug visibility of raw /prices/current rows (channelType included).

Version: v27
- Introduced configurable SELL channel plumbing (pre-work for feed-in support).
- Added internal hooks for SELL advanced pricing (not user-exposed yet).
- Improved fallback behavior when current interval lacks expected fields.
- Additional diagnostics around chosen override slot and label normalization.

Version: v26
- Initial groundwork for separating BUY vs SELL price sources.
- Internal refactors to support future feed-in and advanced SELL pricing.
- Minor safety improvements around missing spotPerKwh values.
- No functional behavior change for default configurations.

Version: v25
- Fixed "feedin" price responses from Amber API (use "general prices")

Version: v24
- Midnight label normalization in final series & target labels (…-24:00).
- Hardened _lookup_price_by_label() to normalize both sides and tolerate spacing/00:00 vs 24:00.
- Index fallback when label not found so overrides never silently miss.
- “Successful override” log lines with provenance, e.g.:
  [amber2sigen] Overrode 23:30-24:00 BUY: 53.02 → 59.18 (from 5-min CurrentInterval)

Notes retained from v23:
- /prices/current requests 5-min first (previous=1,next=1) then falls back to 30-minute.
- Active-slot seeding preference is Current > Forecast > Actual.
- BUY uses --advanced-price (low|predicted|high) if present, else perKwh (falls back as needed).
- SELL uses spotPerKwh (or advancedPrice.<field> for feedIn if --sell-advanced-price / SELLADVANCED enabled).
- Alignment and rotate/canonicalize behavior otherwise unchanged.
- Safety: skip POST if BUY has 0.0 unless --allow-zero-buy.
- Sigen OAuth via encrypted password + token cache.

Examples:
  python3 amber_to_sigen24.py \
    --station-id 92025781200321 \
    --tz Australia/Adelaide \
    --interval 30 \
    --align end \
    --advanced-price predicted \
    --use-current \
    --dry-run
"""

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

AMBER_BASE = "https://api.amber.com.au/v1"
SIGEN_TOKEN_URL = "https://api-aus.sigencloud.com/auth/oauth/token"
SIGEN_SAVE_URL_DEFAULT = "https://api-aus.sigencloud.com/device/stationelecsetprice/save"

# ---- Zero diagnostics buckets (BUY only) ----
ZERO_EVENTS_BUY: List[str] = []  # e.g., "forecast 22:30-23:00", "current 23:00-23:30", "postbuild 01:00-01:30"

# ---------------- Helper label normalizers (v24) ----------------

def _hhmm(dtobj: dt.datetime) -> str:
    return dtobj.strftime("%H:%M")


def _norm_label(s: str) -> str:
    """
    Normalize 'HH:MM-HH:MM' labels:
    - strip spaces
    - map '-00:00' (end) to '-24:00'
    - zero-pad defensively
    """
    s = (s or "").strip().replace(" ", "")
    if s.endswith("-00:00"):
        s = s[:-5] + "-24:00"
    try:
        a, b = s.split("-", 1)
        ah, am = a.split(":"); bh, bm = b.split(":")
        s = f"{int(ah):02d}:{int(am):02d}-{int(bh):02d}:{int(bm):02d}"
    except Exception:
        pass
    return s


def _label_from_range(start_local: dt.datetime, end_local: dt.datetime) -> str:
    """
    Build human label 'HH:MM-HH:MM' with midnight normalized to '24:00' when end rolls to next day 00:00.
    """
    start_s = _hhmm(start_local)
    end_s = _hhmm(end_local)
    if end_s == "00:00" and end_local.date() != start_local.date():
        end_s = "24:00"
    return f"{start_s}-{end_s}"


def _half_hour_index_from_label(label: str) -> Optional[int]:
    """Return 0..47 index based on the *start* HH:MM of the label."""
    try:
        start = _norm_label(label).split("-", 1)[0]
        hh, mm = [int(x) for x in start.split(":")]
        if hh == 24 and mm == 0:  # clamp defensive: 24:00 start should not exist; treat as previous bin start
            hh, mm = 23, 30
        return (hh * 60 + mm) // 30
    except Exception:
        return None


# ---------------- Amber helpers ----------------

def get_site_id(token: str) -> str:
    """Return the first ACTIVE site id for the Amber account (fallback to first)."""
    r = requests.get(f"{AMBER_BASE}/sites", headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    sites = r.json()
    if not sites:
        raise RuntimeError("Amber returned no sites for your token.")
    site = next((s for s in sites if s.get("status") == "ACTIVE"), sites[0])
    return site["id"]


def fetch_amber_prices(token: str, site_id: str, start_date: str, end_date: str,
                       resolution_minutes: int) -> List[dict]:
    params = {"startDate": start_date, "endDate": end_date, "resolution": str(resolution_minutes)}
    r = requests.get(f"{AMBER_BASE}/sites/{site_id}/prices", params=params,
                     headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "intervals" in data:
        rows = data["intervals"]
    else:
        rows = data if isinstance(data, list) else []
    # NOTE: Do NOT filter here; we split later into BUY (general) and SELL (configurable).
    return rows


def fetch_amber_current_triplet_prefer5(token: str, site_id: str) -> Optional[List[dict]]:
    """
    Fetch current & immediate-next rows from /prices/current.
    ALWAYS try 5-minute first (best fidelity: previous=1,next=1), then fall back to 30-minute.
    Returns a list (1–2 rows) or None.
    """
    url = f"{AMBER_BASE}/sites/{site_id}/prices/current"

    def _get(res):
        params = {"previous": "1", "next": "1", "resolution": str(res)}
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            rows = data
        elif isinstance(data, dict) and data:
            rows = [data]
        else:
            return None
        # NOTE: Do NOT filter here; trip may include both general + feedIn.
        return rows

    trip = _get(5)
    if not trip:
        trip = _get(30)
    return trip


# ---------------- Slot building & mapping (UTC-normalized) ----------------

def floor_to_step(ts: dt.datetime, step_min: int) -> dt.datetime:
    minute = (ts.minute // step_min) * step_min
    return ts.replace(second=0, microsecond=0, minute=minute)


def build_window(now_local: dt.datetime, step_min: int, total_minutes: int = 1440) -> List[Tuple[dt.datetime, dt.datetime]]:
    """Build a list of [start,end) UTC time ranges covering now -> now+24h in step_min steps."""
    start_local = floor_to_step(now_local, step_min)
    start_utc = start_local.astimezone(dt.timezone.utc)
    out: List[Tuple[dt.datetime, dt.datetime]] = []
    t = start_utc
    end = start_utc + dt.timedelta(minutes=total_minutes)
    step = dt.timedelta(minutes=step_min)
    while t < end:
        out.append((t, t + step))  # UTC
        t += step
    return out


def parse_iso_utc(s: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d.astimezone(dt.timezone.utc)


def index_prices_by_start_utc(amber_rows: List[dict], step_min: int) -> Dict[dt.datetime, dict]:
    m: Dict[dt.datetime, dict] = {}
    for row in amber_rows:
        st = row.get("startTime")
        if not st:
            continue
        try:
            t = parse_iso_utc(st)
            t = floor_to_step(t, step_min)  # squash any ':01Z' drift
            m[t] = row
        except Exception:
            continue
    return m


def index_prices_by_end_utc(amber_rows: List[dict], step_min: int) -> Dict[dt.datetime, dict]:
    m: Dict[dt.datetime, dict] = {}
    for row in amber_rows:
        et = row.get("endTime")
        if not et:
            continue
        try:
            t = parse_iso_utc(et)
            t = floor_to_step(t, step_min)
            m[t] = row
        except Exception:
            continue
    return m


def get_value(row: Optional[dict], key: str) -> Optional[float]:
    """Return float value for 'key'; supports dotted paths e.g. 'advancedPrice.predicted'."""
    if not row:
        return None
    try:
        if "." in key:
            outer, inner = key.split(".", 1)
            v = row.get(outer, {}).get(inner)
        else:
            v = row.get(key)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


# ---------------- Baseline / carry-forward helpers ----------------

def last_known_before(
    rows: List[dict],
    key: str,
    step_min: int,
    align: str,
    now_utc: dt.datetime
) -> Optional[float]:
    """Find the most recent available value strictly before 'now' (UTC) using chosen alignment."""
    index = index_prices_by_end_utc(rows, step_min) if align == "end" \
        else index_prices_by_start_utc(rows, step_min)
    prev_keys = [t for t in index.keys() if t < now_utc]
    if not prev_keys:
        return None
    t = max(prev_keys)
    return get_value(index[t], key)


# ---------------- Series building / rotation / labels ----------------

def build_series_for_window(
    slots_utc: List[Tuple[dt.datetime, dt.datetime]],
    tz: dt.tzinfo,
    rows: List[dict],
    key: str,
    step_min: int,
    align: str = "start",  # "start" or "end"
    initial_last: Optional[float] = None,  # baseline to avoid leading zeros / tail gaps
) -> List[Tuple[str, float]]:
    if align == "end":
        by_key = index_prices_by_end_utc(rows, step_min)
    else:
        by_key = index_prices_by_start_utc(rows, step_min)

    out: List[Tuple[str, float]] = []
    last = 0.0 if initial_last is None else float(initial_last)
    for (t0_utc, t1_utc) in slots_utc:
        anchor = t0_utc if align == "start" else t1_utc
        row = by_key.get(anchor)
        val = get_value(row, key) if row else None
        if val is not None:
            last = val
        t0_local = t0_utc.astimezone(tz)
        t1_local = t1_utc.astimezone(tz)
        # label here is interim; canonicalize later to day (adds 24:00 on last bin)
        out.append((f"{t0_local.strftime('%H:%M')}-{t1_local.strftime('%H:%M')}", round(last, 2)))
    return out


def rotate_series_to_midnight(series: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    prefix = "00:00-"
    idx00 = next((i for i, (tr, _) in enumerate(series) if tr.startswith(prefix)), None)
    if idx00 is None:
        return series
    return series[idx00:] + series[:idx00]


def canonicalize_series_to_day(series: List[Tuple[str, float]], step_min: int) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for i, (_, price) in enumerate(series):
        start_min = i * step_min
        end_min = (i + 1) * step_min
        sh, sm = divmod(start_min, 60)
        eh, em = divmod(end_min, 60)
        s_lbl = f"{sh:02d}:{sm:02d}"
        e_lbl = "24:00" if end_min == 1440 else f"{eh:02d}:{em:02d}"
        out.append((f"{s_lbl}-{e_lbl}", price))
    return out


def shift_series(series: List[Tuple[str, float]], slots: int) -> List[Tuple[str, float]]:
    """Rotate series by N slots after pricing (positive = later, negative = earlier)."""
    if not slots:
        return series
    dq = deque(series)
    dq.rotate(slots)
    return list(dq)


def _lookup_price_by_label(series: List[Tuple[str, float]], label: str) -> Optional[float]:
    """
    Find price by human label; robust to '-00:00' vs '-24:00', stray spaces, zero-padding.
    Also tries a start-time prefix match if exact label not present. (v24-hardened)
    """
    want = _norm_label(label)
    # 1) exact normalized match
    for tr, p in series:
        if _norm_label(tr) == want:
            return p
    # 2) start-time prefix fallback
    try:
        want_start = want.split("-", 1)[0]
        for tr, p in series:
            if _norm_label(tr).split("-", 1)[0] == want_start:
                return p
    except Exception:
        pass
    return None


# ---------------- Sigen OAuth helpers (supports SIGEN_PASS and SIGEN_PASS_ENC) ----------------

# Sigenergy password encryption constants
_SIGENERGY_AES_KEY = b"sigensigensigenp"  # 16 bytes for AES-128
_SIGENERGY_AES_IV = b"sigensigensigenp"   # Same as key


def encode_sigenergy_password(plain_password: str) -> str:
    """Encode a plain password to Sigenergy's encrypted format.

    Sigenergy uses AES-128-CBC with PKCS7 padding, then Base64 encodes the result.
    Key and IV are both "sigensigensigenp".

    Args:
        plain_password: The plain text password

    Returns:
        Base64-encoded encrypted password (pass_enc format)
    """
    cipher = AES.new(_SIGENERGY_AES_KEY, AES.MODE_CBC, _SIGENERGY_AES_IV)
    padded_data = pad(plain_password.encode("utf-8"), AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return base64.b64encode(encrypted).decode("utf-8")


def cache_path_for(user: str) -> str:
    base = os.path.join(os.path.expanduser("~"), ".cache", "amber_to_sigen")
    os.makedirs(base, exist_ok=True)
    key = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, f"sigen_{key}.json")


def load_cached_tokens(user: str):
    try:
        p = cache_path_for(user)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cached_tokens(user: str, tok: dict):
    p = cache_path_for(user)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tok, f)


def token_from_response(j: dict) -> dict:
    if not isinstance(j, dict):
        raise RuntimeError(f"Sigen token error: non-JSON/unknown body: {repr(j)[:400]}")

    data = j.get("data")
    if isinstance(data, dict) and ("access_token" in data or "token" in data):
        access = data.get("access_token") or data.get("token")
        refresh = data.get("refresh_token", "")
        ttype = data.get("token_type", "Bearer")
        expires_in = int(data.get("expires_in", 3600))
        return {"access_token": access, "refresh_token": refresh, "token_type": ttype,
                "expires_at": time.time() + expires_in - 60}

    if "access_token" in j or "token" in j:
        access = j.get("access_token") or j.get("token")
        refresh = j.get("refresh_token", "")
        ttype = j.get("token_type", "Bearer")
        expires_in = int(j.get("expires_in", 3600))
        return {"access_token": access, "refresh_token": refresh, "token_type": ttype,
                "expires_at": time.time() + expires_in - 60}

    code = j.get("code")
    msg = j.get("msg") or j.get("error_description") or j.get("error") or "unknown"
    raise RuntimeError(f"Sigen token error: code={code} msg={msg} body={json.dumps(j)[:400]}")


def sigen_password_grant_encrypted(username: str, enc_password_b64: str, user_device_id: str) -> dict:
    headers = {
        "Authorization": "Basic c2lnZW46c2lnZW4=",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "username": username,
        "password": enc_password_b64,
        "scope": "server",
        "grant_type": "password",
        "userDeviceId": user_device_id
    }
    r = requests.post(SIGEN_TOKEN_URL, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return token_from_response(r.json())


def sigen_refresh(refresh_token: str) -> dict:
    headers = {
        "Authorization": "Basic c2lnZW46c2lnZW4=",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    r = requests.post(SIGEN_TOKEN_URL, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return token_from_response(r.json())


def ensure_sigen_headers(user: str, user_device_id: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if not user:
        raise RuntimeError("SIGEN_USER is required (env or --sigen-user).")

    cached = load_cached_tokens(user)
    now = time.time()
    if cached and cached.get("expires_at", 0) > now and cached.get("access_token"):
        headers["Authorization"] = f"{cached.get('token_type','Bearer')} {cached['access_token']}"
        return headers

    if cached and cached.get("refresh_token"):
        try:
            newtok = sigen_refresh(cached["refresh_token"])
            save_cached_tokens(user, newtok)
            headers["Authorization"] = f"{newtok.get('token_type','Bearer')} {newtok['access_token']}"
            return headers
        except Exception:
            pass

    # Support both plain password (SIGEN_PASS) and pre-encoded password (SIGEN_PASS_ENC)
    # Priority: SIGEN_PASS_ENC (explicit) > SIGEN_PASS (will be encoded)
    enc_pw = os.environ.get("SIGEN_PASS_ENC")
    plain_pw = os.environ.get("SIGEN_PASS")

    if enc_pw:
        # Use pre-encoded password directly
        newtok = sigen_password_grant_encrypted(user, enc_pw, user_device_id)
        save_cached_tokens(user, newtok)
        headers["Authorization"] = f"{newtok.get('token_type','Bearer')} {newtok['access_token']}"
        return headers

    if plain_pw:
        # Encode plain password and use it
        enc_pw = encode_sigenergy_password(plain_pw)
        newtok = sigen_password_grant_encrypted(user, enc_pw, user_device_id)
        save_cached_tokens(user, newtok)
        headers["Authorization"] = f"{newtok.get('token_type','Bearer')} {newtok['access_token']}"
        return headers

    raise RuntimeError(
        "No way to authenticate: set SIGEN_PASS (plain password) or SIGEN_PASS_ENC (encrypted password), "
        "along with SIGEN_DEVICE_ID."
    )


# ---------------- Utilities ----------------

def parse_boolish_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------- Main flow ----------------

def main():
    ap = argparse.ArgumentParser(description="Amber -> Sigen staticPricing (now->+24h)")
    ap.add_argument("--amber-token", default=os.environ.get("AMBER_TOKEN"))
    ap.add_argument("--site-id")
    ap.add_argument("--tz", default="Australia/Adelaide")
    ap.add_argument("--interval", type=int, default=int(os.environ.get("INTERVAL", "30")),
                    choices=[5, 30], help="Slot size & Amber resolution (minutes)")
    ap.add_argument("--align", default="end", choices=["start", "end"],
                    help="Align Amber price rows to slot start or end when labeling (Amber app tends to be 'end').")
    ap.add_argument("--slot-shift", type=int, default=0,
                    help="Rotate series by N slots after pricing (positive=later, negative=earlier).")
    ap.add_argument("--advanced-price", choices=["low", "predicted", "high"],
                    help="Use advancedPrice.<field> instead of perKwh for BUY price.")

    # Default USE_CURRENT = True unless explicitly disabled via env or CLI
    env_use_current = parse_boolish_env("USE_CURRENT", True)
    ap.add_argument("--use-current", dest="use_current", action="store_true", default=env_use_current)
    ap.add_argument("--no-use-current", dest="use_current", action="store_false",
                    help="Disable /prices/current override of active slot.")

    # SELL channel selection (default: general). Env SELL_CHANNEL=general|feedIn
    ap.add_argument("--sell-channel",
                    default=(os.environ.get("SELL_CHANNEL", "general").strip().lower()),
                    choices=["general", "feedin"],
                    help="Which Amber channelType to use for SELL: general (default) or feedIn.")

    # --- NEW: SELL advanced price selection (feedIn only). Env SELLADVANCED=low|predicted|high
    ap.add_argument("--sell-advanced-price",
                    default=(os.environ.get("SELLADVANCED", "").strip().lower() or None),
                    choices=["low", "predicted", "high"],
                    help="Use advancedPrice.<field> for SELL when --sell-channel feedIn. "
                         "Amber returns feed-in advanced values negative; we flip sign to positive.")

    ap.add_argument("--station-id", type=int, required=True)
    ap.add_argument("--plan-name", default="SAPN TOU")
    ap.add_argument("--sigen-url", default=os.environ.get("SIGEN_SAVE_URL", SIGEN_SAVE_URL_DEFAULT))
    ap.add_argument("--sigen-user", default=os.environ.get("SIGEN_USER"))
    ap.add_argument("--device-id", default=os.environ.get("SIGEN_DEVICE_ID", "1756353655250"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-zero-buy", action="store_true",
                    help="Allow POST even if final BUY contains 0.0 (unsafe).")

    args = ap.parse_args()
    if not args.amber_token:
        ap.error("Missing Amber token (env AMBER_TOKEN or --amber-token).")

    step_min = args.interval

    # Time zone
    try:
        import zoneinfo  # Python 3.9+
        tz = zoneinfo.ZoneInfo(args.tz)
    except Exception:
        tz = dt.datetime.now().astimezone().tzinfo

    now_local = dt.datetime.now(tz)
    now_utc = dt.datetime.now(dt.timezone.utc)

    # Build slots for the next 24h window from 'now'
    slots = build_window(now_local, step_min=step_min, total_minutes=1440)

    # Human label for the "active" slot from now_local (normalize midnight end to 24:00)
    active_start_local = floor_to_step(now_local, step_min)
    active_end_local = active_start_local + dt.timedelta(minutes=step_min)
    current_label = _label_from_range(active_start_local, active_end_local)

    # Fetch Amber bulk prices today + tomorrow
    today = now_local.date().strftime("%Y-%m-%d")
    tomorrow = (now_local.date() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    site_id = args.site_id or get_site_id(args.amber_token)

    rows = fetch_amber_prices(args.amber_token, site_id,
                              start_date=today, end_date=tomorrow,
                              resolution_minutes=step_min)

    # Split by channelType: BUY is always "general", SELL is configurable
    def _ctype(r): return (r.get("channelType") or "").strip().lower()
    sell_ct = args.sell_channel.strip().lower()
    buy_rows = [r for r in rows if _ctype(r) == "general"]
    sell_rows = [r for r in rows if _ctype(r) == sell_ct]

    # --- NEW: SELL advanced toggle + key/transform
    sell_uses_adv = (sell_ct == "feedin" and bool(args.sell_advanced_price))
    sell_key = (f"advancedPrice.{args.sell_advanced_price}" if sell_uses_adv else "spotPerKwh")

    # Keys and baselines (carry forward if missing)
    buy_key = f"advancedPrice.{args.advanced_price}" if args.advanced_price else "perKwh"
    buy_baseline = last_known_before(buy_rows, buy_key, step_min, args.align, now_utc)
    sell_baseline = last_known_before(sell_rows, sell_key, step_min, args.align, now_utc)
    # Flip sign for feed-in advanced baseline (Amber advanced feed-in is negative)
    if sell_uses_adv and sell_baseline is not None:
        sell_baseline = -float(sell_baseline)

    # Build series with baseline (prevents /0.0 at head/tail if data isn't published yet)
    buy_ranges = build_series_for_window(slots, tz, buy_rows, key=buy_key,
                                         step_min=step_min, align=args.align,
                                         initial_last=buy_baseline)
    sell_ranges = build_series_for_window(slots, tz, sell_rows, key=sell_key,
                                          step_min=step_min, align=args.align,
                                          initial_last=sell_baseline)
    # --- NEW: Flip sign if using feed-in advanced (make positive for SELL)
    if sell_uses_adv:
        sell_ranges = [(tr, round(-float(p), 2)) for tr, p in sell_ranges]

    # Optional fine-tune shift (after values are picked)
    buy_ranges = shift_series(buy_ranges, args.slot_shift)
    sell_ranges = shift_series(sell_ranges, args.slot_shift)

    # Diagnostics: record any zeros from forecast build (BUY only)
    for tr, p in buy_ranges:
        if p == 0.0:
            ZERO_EVENTS_BUY.append(f"forecast {tr}")

    # ---- Compute a pending override target from /prices/current (prefer 5-min, fallback 30-min) ----
    pending_override = None  # (target_label, buy_value, sell_value)

    if args.use_current:
        trip = fetch_amber_current_triplet_prefer5(args.amber_token, site_id)

        def pick_buy(row: dict) -> Optional[float]:
            if not row:
                return None
            if args.advanced_price:
                adv = row.get("advancedPrice") or {}
                v = adv.get(args.advanced_price)
                if v is not None:
                    return float(v)
            v = row.get("perKwh")
            return float(v) if v is not None else None

#        # --- UPDATED: pick_sell respects SELLADVANCED for feedIn and flips sign
         # --- UPDATED: pick_sell prefers feedIn advanced (invert), then feedIn spot, else None
        def pick_sell(row: dict) -> Optional[float]:
            if not row:
                return None
            if sell_uses_adv:
                adv = row.get("advancedPrice") or {}
                v = adv.get(args.sell_advanced_price)
#                if v is None:
#                    return None
#                return -float(v)  # flip sign: Amber advanced feed-in is negative
                if v is not None:
                    return -float(v)  # always invert advanced feed-in
                # NEW: if advanced missing, use feedIn spotPerKwh (no inversion)
                v2 = row.get("spotPerKwh")
                return float(v2) if v2 is not None else None
            v = row.get("spotPerKwh")
            return float(v) if v is not None else None

        def local_start_end(row: dict) -> Optional[Tuple[dt.datetime, dt.datetime, str]]:
            try:
                st_l = parse_iso_utc(row["startTime"]).astimezone(tz)
                en_l = parse_iso_utc(row["endTime"]).astimezone(tz)
                lbl = _label_from_range(st_l, en_l)  # v24 normalize midnight
                return st_l, en_l, lbl
            except Exception:
                return None

        # --- DEBUG: dump raw triplet key fields ---
        if trip:
            try:
                dbg_rows = []
                for r in trip:
                    t = r.get("type")
                    st = r.get("startTime")
                    en = r.get("endTime")
                    per = r.get("perKwh")
                    spot = r.get("spotPerKwh")
                    adv = r.get("advancedPrice")
                    dbg_rows.append({"type": t, "channelType": r.get("channelType"),
                                     "startTime": st, "endTime": en,
                                     "perKwh": per, "spotPerKwh": spot, "advancedPrice": adv})
                print("[amber2sigen] /prices/current triplet (raw key fields):", file=sys.stderr)
                print(json.dumps(dbg_rows, indent=2), file=sys.stderr)
            except Exception:
                pass

        # Rank rows: Current > Forecast > Actual (closest we’ll get to “right now”)
        def rank_type(t: str) -> int:
            if t == "CurrentInterval": return 0
            if t == "ForecastInterval": return 1
            return 2  # ActualInterval last

        chosen = None

        if trip:
            # Partition by channel
            trip_buy = [r for r in trip if (r.get("channelType") or "").strip().lower() == "general"]
            trip_sell = [r for r in trip if (r.get("channelType") or "").strip().lower() == sell_ct]

            # Prefer the same-type row whose START is closest to active slot start.
            def _dist(row):
                try:
                    st = parse_iso_utc(row["startTime"]).astimezone(tz)
                    return abs((st - active_start_local).total_seconds())
                except Exception:
                    return 10**9

            # Choose BUY row (from 'general')
            chosen_buy_row = None
            if trip_buy:
                trip_buy_sorted = sorted(trip_buy, key=lambda r: (rank_type(r.get("type","")), _dist(r)))
                chosen_buy_row = trip_buy_sorted[0]

            chosen = chosen_buy_row

            if chosen:
                se = local_start_end(chosen)
                chosen_buy = pick_buy(chosen)

                # Determine target label (same logic as before)
                if se:
                    st_l, en_l, lbl = se
                    if step_min == 30:
                        # Enclosing 30-min slot label for the chosen 5-min interval (normalize midnight)
                        slot_start = floor_to_step(st_l, 30)
                        slot_end = slot_start + dt.timedelta(minutes=30)
                        target_label = _label_from_range(slot_start, slot_end)
                    else:
                        # step_min == 5 → the 5-min label itself (normalized)
                        target_label = lbl

                    # SELL: pick row from chosen SELL channel for the same slot
                    chosen_sell = None
                    if trip_sell:
                        def _label_for(row):
                            st_l2 = parse_iso_utc(row["startTime"]).astimezone(tz)
                            en_l2 = parse_iso_utc(row["endTime"]).astimezone(tz)
                            if step_min == 30:
                                st_l2 = floor_to_step(st_l2, 30)
                                en_l2 = st_l2 + dt.timedelta(minutes=30)
                            return _label_from_range(st_l2, en_l2)

                        # Try exact normalized label match first
                        sell_match = next((r for r in trip_sell
                                           if _norm_label(_label_for(r)) == _norm_label(target_label)), None)
                        if not sell_match:
                            # Fallback: closest start time by the same ranking
                            trip_sell_sorted = sorted(trip_sell, key=lambda r: (rank_type(r.get("type","")), _dist(r)))
                            sell_match = trip_sell_sorted[0]
#                        chosen_sell = pick_sell(sell_match) if sell_match else None
                        chosen_sell = pick_sell(sell_match) if sell_match else None

                    # If SELLADVANCED is on and feedIn lacks advanced for this slot, fallback to GENERAL spotPerKwh
                    if chosen_sell is None and sell_uses_adv and trip_buy:
                        # nearest BUY/general row by the same ranking & distance
                        fb_buy = sorted(trip_buy, key=lambda r: (rank_type(r.get("type","")), _dist(r)))[0]
                        v = fb_buy.get("spotPerKwh")
                        chosen_sell = float(v) if v is not None else None

                    pending_override = (target_label, chosen_buy, chosen_sell)

#                    # Human-friendly summary (reflect SELLADVANCED if enabled)
#                    labels, buy_vals, sell_vals = [], [], []
#                    for r in trip:
#                        try:
#                            st = parse_iso_utc(r["startTime"]).astimezone(tz)
#                            en = parse_iso_utc(r["endTime"]).astimezone(tz)
#                            lbl2 = _label_from_range(st, en)  # normalized
#                        except Exception:
#                            lbl2 = ""
#                        # BUY
#                        if args.advanced_price:
#                            adv = (r.get("advancedPrice") or {})
#                            b = adv.get(args.advanced_price)
#                            if b is None:
#                                b = r.get("perKwh")
#                        else:
#                            b = r.get("perKwh")
#                        # SELL (spot or flipped advanced)
#                        if sell_uses_adv:
#                            advs = (r.get("advancedPrice") or {})
#                            s = advs.get(args.sell_advanced_price)
#                            s = None if s is None else round(-float(s), 2)
#                        else:
#                            s = r.get("spotPerKwh")
#                            s = None if s is None else round(float(s), 2)
#                        labels.append(lbl2)
#                        buy_vals.append(None if b is None else round(float(b), 2))
#                        sell_vals.append(s)
#
#                    print("[amber2sigen] Current window BUY slots = " +
#                          ", ".join(f"{l}:{('' if b is None else b)}" for l, b in zip(labels, buy_vals)),
#                          file=sys.stderr)
#                    print("[amber2sigen] Current window SELL slots = " +
#                          ", ".join(f"{l}:{('' if s is None else s)}" for l, s in zip(labels, sell_vals)),
#                          file=sys.stderr)

                    # Human-friendly summary per channel (reflect SELLADVANCED if enabled)
                    def _label_of(row):
                        st = parse_iso_utc(row["startTime"]).astimezone(tz)
                        en = parse_iso_utc(row["endTime"]).astimezone(tz)
                        return _label_from_range(st, en)  # normalized

                    buy_labels, buy_vals = [], []
                    for r in trip_buy:
                        lbl2 = _label_of(r)
                        b = pick_buy(r)
                        buy_labels.append(lbl2)
                        buy_vals.append("" if b is None else round(float(b), 2))

                    sell_labels, sell_vals = [], []
                    for r in trip_sell:
                        lbl2 = _label_of(r)
                        s = pick_sell(r)  # already flips sign if SELLADVANCED is set
                        sell_labels.append(lbl2)
                        sell_vals.append("" if s is None else round(float(s), 2))

                    buy_adv_lbl  = args.advanced_price or "off"
                    sell_adv_lbl = (args.sell_advanced_price if sell_uses_adv else "off")
                    print(f"[amber2sigen] Current window BUY slots [ch=general adv={buy_adv_lbl}] = " +
                          ", ".join(f"{l}:{v}" for l, v in zip(buy_labels, buy_vals)),
                          file=sys.stderr)
                    print(f"[amber2sigen] Current window SELL slots [ch={sell_ct} adv={sell_adv_lbl}] = " +
                          ", ".join(f"{l}:{v}" for l, v in zip(sell_labels, sell_vals)),
                          file=sys.stderr) 

#                   # Annotate with channel + Advanced level (or 'off')
#                    buy_adv_lbl  = args.advanced_price or "off"
#                    sell_adv_lbl = (args.sell_advanced_price if sell_uses_adv else "off")
#                    print(f"[amber2sigen] Current window BUY slots [ch=general adv={buy_adv_lbl}] = " +
#                          ", ".join(f"{l}:{('' if b is None else b)}" for l, b in zip(labels, buy_vals)),
#                          file=sys.stderr)
#                    print(f"[amber2sigen] Current window SELL slots [ch={sell_ct} adv={sell_adv_lbl}] = " +
#                          ", ".join(f"{l}:{('' if s is None else s)}" for l, s in zip(labels, sell_vals)),
#                          file=sys.stderr)

                else:
                    print("[amber2sigen] /current row lacked parseable start/end; skipping override.", file=sys.stderr)
            else:
                print("[amber2sigen] No suitable /current BUY row found (channel=general).", file=sys.stderr)

    # ---- Rotate + canonicalize to midnight day labels BEFORE applying override ----
    buy_ranges  = rotate_series_to_midnight(buy_ranges)
    sell_ranges = rotate_series_to_midnight(sell_ranges)
    buy_ranges  = canonicalize_series_to_day(buy_ranges, step_min)
    sell_ranges = canonicalize_series_to_day(sell_ranges, step_min)

    # ---- Apply pending override by label with normalization + index fallback (v24) ----
    if pending_override:
        target_label, chosen_buy, chosen_sell = pending_override

        # --- BUY ---
        buy_idx = next((i for i, (tr, _) in enumerate(buy_ranges)
                        if _norm_label(tr) == _norm_label(target_label)
                        or _norm_label(tr).split("-", 1)[0] == _norm_label(target_label).split("-", 1)[0]), None)
        if buy_idx is None:
            print(f"[amber2sigen] Could not locate BUY target label {target_label} in final series.", file=sys.stderr)
            buy_idx = _half_hour_index_from_label(target_label) if step_min == 30 else None
            if buy_idx is None or not (0 <= buy_idx < len(buy_ranges)):
                print(f"[amber2sigen] FATAL: cannot compute index for BUY label {target_label}; no override applied.",
                      file=sys.stderr)
            else:
                idx_label, idx_old = buy_ranges[buy_idx]
                if chosen_buy is None or chosen_buy == 0.0:
                    ZERO_EVENTS_BUY.append(f"current {idx_label}")
                    print(f"[amber2sigen] Skipping BUY current override (zero/missing) for {idx_label}; "
                          f"keeping forecast {idx_old}", file=sys.stderr)
                else:
                    buy_new = round(float(chosen_buy), 2)
                    buy_ranges[buy_idx] = (idx_label, buy_new)
                    print(f"[amber2sigen] Overrode {idx_label} BUY: {idx_old:.2f} → {buy_new:.2f} "
                          f"(from 5-min CurrentInterval)", file=sys.stderr)
        else:
            old = buy_ranges[buy_idx][1]
            if chosen_buy is None or chosen_buy == 0.0:
                ZERO_EVENTS_BUY.append(f"current {buy_ranges[buy_idx][0]}")
                print(f"[amber2sigen] Skipping BUY current override (zero/missing) for "
                      f"{buy_ranges[buy_idx][0]}; keeping forecast {old}", file=sys.stderr)
            else:
                buy_new = round(float(chosen_buy), 2)
                lbl = buy_ranges[buy_idx][0]
                buy_ranges[buy_idx] = (lbl, buy_new)
                print(f"[amber2sigen] Overrode {lbl} BUY: {old:.2f} → {buy_new:.2f} "
                      f"(from 5-min CurrentInterval)", file=sys.stderr)

        # --- SELL ---
        sell_idx = next((i for i, (tr, _) in enumerate(sell_ranges)
                         if _norm_label(tr) == _norm_label(target_label)
                         or _norm_label(tr).split("-", 1)[0] == _norm_label(target_label).split("-", 1)[0]), None)
        if sell_idx is None:
            print(f"[amber2sigen] Could not locate SELL target label {target_label} in final series.", file=sys.stderr)
            sell_idx = _half_hour_index_from_label(target_label) if step_min == 30 else None
            if sell_idx is None or not (0 <= sell_idx < len(sell_ranges)):
                print(f"[amber2sigen] FATAL: cannot compute index for SELL label {target_label}; no override applied.",
                      file=sys.stderr)
            else:
                idx_label, idx_old = sell_ranges[sell_idx]
                if chosen_sell is None:
                    print(f"[amber2sigen] Skipping SELL current override (missing) for {idx_label}; "
                          f"keeping forecast {idx_old}", file=sys.stderr)
                else:
                    sell_new = round(float(chosen_sell), 2)
                    sell_ranges[sell_idx] = (idx_label, sell_new)
                    print(f"[amber2sigen] Overrode {idx_label} SELL: {idx_old:.2f} → {sell_new:.2f} "
                          f"(from 5-min CurrentInterval)", file=sys.stderr)
        else:
            old = sell_ranges[sell_idx][1]
            if chosen_sell is None:
                print(f"[amber2sigen] Skipping SELL current override (missing) for "
                      f"{sell_ranges[sell_idx][0]}; keeping forecast {old}", file=sys.stderr)
            else:
                sell_new = round(float(chosen_sell), 2)
                lbl = sell_ranges[sell_idx][0]
                sell_ranges[sell_idx] = (lbl, sell_new)
                print(f"[amber2sigen] Overrode {lbl} SELL: {old:.2f} → {sell_new:.2f} "
                      f"(from 5-min CurrentInterval)", file=sys.stderr)

    # Postbuild zero scan (final series)
    for tr, p in buy_ranges:
        if p == 0.0:
            ZERO_EVENTS_BUY.append(f"postbuild {tr}")

    # End-of-run diagnostics: active slot BUY/SELL in final series (normalized lookup)
    print(f"[amber2sigen] Active slot BUY {current_label} = "
          f"{_lookup_price_by_label(buy_ranges, current_label) or ''}", file=sys.stderr)
    print(f"[amber2sigen] Active slot SELL {current_label} = "
          f"{_lookup_price_by_label(sell_ranges, current_label) or ''}", file=sys.stderr)

    # Summarize zero events
    if ZERO_EVENTS_BUY:
        uniq = sorted(set(ZERO_EVENTS_BUY))
        print(f"[amber2sigen] BUY perKwh saw 0.0 in: {', '.join(uniq)}", file=sys.stderr)
    else:
        print("[amber2sigen] BUY perKwh: no 0.0 placeholders detected.", file=sys.stderr)

    # Build payload
    payload = {
        "stationId": args.station_id,
        "priceMode": 1,
        "buyPrice": {
            "dynamicPricing": None,
            "staticPricing": {
                "providerName": "Amber",
                "tariffCode": "",
                "tariffName": "",
                "currencyCode": "Cent",
                "subAreaName": "",
                "planName": f"{args.plan_name} {step_min}-min",
                "combinedPrices": [
                    {
                        "monthRange": "01-12",
                        "weekPrices": [
                            {
                                "weekRange": "1-7",
                                "timeRange": [{"timeRange": tr, "price": p} for tr, p in buy_ranges]
                            }
                        ]
                    }
                ]
            }
        },
        "sellPrice": {
            "dynamicPricing": None,
            "staticPricing": {
                "providerName": "Amber",
                "tariffCode": "",
                "tariffName": "",
                "currencyCode": "Cent",
                "subAreaName": "",
                "planName": f"{args.plan_name} {step_min}-min",
                "combinedPrices": [
                    {
                        "monthRange": "01-12",
                        "weekPrices": [
                            {
                                "weekRange": "1-7",
                                "timeRange": [{"timeRange": tr, "price": p} for tr, p in sell_ranges]
                            }
                        ]
                    }
                ]
            }
        }
    }

    headers = ensure_sigen_headers(args.sigen_user, args.device_id)

    # Show what we will send (controlled by PAYLOAD_DEBUG: 0=off, 1=on)
    if os.environ.get("PAYLOAD_DEBUG", "1").strip() == "1":
        print(json.dumps(payload, indent=2))
    else:
        print("[amber2sigen] PAYLOAD_DEBUG=0 → payload print suppressed.", file=sys.stderr)

    # Final safety: only skip POST if final BUY contains a 0.0 (unless override)
    final_has_zero = any(p == 0.0 for _, p in buy_ranges)
    if args.dry_run or (final_has_zero and not args.allow_zero_buy):
        if final_has_zero and not args.allow_zero_buy:
            print("[amber2sigen] Final BUY series still contains 0.0 → skipping POST to Sigen.", file=sys.stderr)
        else:
            print("[amber2sigen] --dry-run set → skipping POST to Sigen.", file=sys.stderr)
        return

    # POST
    r = requests.post(args.sigen_url, headers=headers, json=payload, timeout=60)
    try:
        r.raise_for_status()
    except Exception:
        print(f"Error from Sigen: HTTP {r.status_code}\n{r.text}", file=sys.stderr)
        raise
    print("Sigen response:", r.status_code, r.text)


if __name__ == "__main__":
    main()