#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amber (5-minute or 30-minute) -> Sigen staticPricing (now -> +24h)

Version: v25
- Fixed "feedin" price responses from Amber API (use "general prices")

Version: v24
- Midnight label normalization in final series & target labels (…-24:00).
- Hardened _lookup_price_by_label() to normalize both sides and tolerate spacing/00:00 vs 24:00.
- Index fallback when label not found so overrides never silently miss.
- “Successful override” log lines with provenance, e.g.:
  [amber2sigen] Overrode 23:30-24:00 BUY: 53.02 → 59.18 (from 5-min CurrentInterval)

Notes retained from v23:
- /prices/current requests 5-min first (previous=1,next=1) then falls back to 30-min.
- Active-slot seeding preference is Current > Forecast > Actual.
- BUY uses --advanced-price (low|predicted|high) if present, else perKwh (falls back as needed).
- SELL uses spotPerKwh.
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
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

import requests

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
        if hh == 24 and mm == 0:
            # clamp defensive: 24:00 start should not exist; treat as previous bin start
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
    # --- NEW: filter out feedIn etc ---
    rows = [r for r in rows if r.get("channelType", "").lower() == "general"]
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
        # --- NEW: filter channelType ---
        return [row for row in rows if row.get("channelType", "").lower() == "general"]

    trip = _get(5)
    if not trip:
        trip = _get(30)
    return trip


# ---------------- Slot building & mapping (UTC-normalized) ----------------

def floor_to_step(ts: dt.datetime, step_min: int) -> dt.datetime:
    minute = (ts.minute // step_min) * step_min
    return ts.replace(second=0, microsecond=0, minute=minute)

def build_window(now_local: dt.datetime, step_min: int,
                 total_minutes: int = 1440) -> List[Tuple[dt.datetime, dt.datetime]]:
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
    rows: List[dict], key: str, step_min: int, align: str, now_utc: dt.datetime
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
    align: str = "start",             # "start" or "end"
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
    """Find price by human label; robust to '-00:00' vs '-24:00', stray spaces, zero-padding.
    Also tries a start-time prefix match if exact label not present. (v24-hardened)"""
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

# ---------------- Sigen OAuth helpers (supports SIGEN_PASS_ENC) ----------------

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
    enc_pw = os.environ.get("SIGEN_PASS_ENC")
    if enc_pw:
        newtok = sigen_password_grant_encrypted(user, enc_pw, user_device_id)
        save_cached_tokens(user, newtok)
        headers["Authorization"] = f"{newtok.get('token_type','Bearer')} {newtok['access_token']}"
        return headers
    raise RuntimeError("No way to authenticate: set SIGEN_PASS_ENC (encrypted password) and SIGEN_DEVICE_ID.")

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

    # Keys and baselines (carry forward if missing)
    buy_key = f"advancedPrice.{args.advanced_price}" if args.advanced_price else "perKwh"
    buy_baseline = last_known_before(rows, buy_key, step_min, args.align, now_utc)
    sell_baseline = last_known_before(rows, "spotPerKwh", step_min, args.align, now_utc)

    # Build series with baseline (prevents <nil>/0.0 at head/tail if data isn't published yet)
    buy_ranges = build_series_for_window(slots, tz, rows, key=buy_key,
                                         step_min=step_min, align=args.align,
                                         initial_last=buy_baseline)
    sell_ranges = build_series_for_window(slots, tz, rows, key="spotPerKwh",
                                          step_min=step_min, align=args.align,
                                          initial_last=sell_baseline)

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

        def pick_sell(row: dict) -> Optional[float]:
            if not row:
                return None
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
                    dbg_rows.append({"type": t, "startTime": st, "endTime": en,
                                     "perKwh": per, "spotPerKwh": spot, "advancedPrice": adv})
                print("[amber2sigen] /prices/current triplet (raw key fields):", file=sys.stderr)
                print(json.dumps(dbg_rows, indent=2), file=sys.stderr)
            except Exception:
                pass

            # Rank rows: Current > Forecast > Actual (closest we’ll get to “right now”)
            def rank_type(t: str) -> int:
                if t == "CurrentInterval":
                    return 0
                if t == "ForecastInterval":
                    return 1
                return 2  # ActualInterval last

            chosen = None
            if trip:
                # Prefer the same-type row whose START is closest to active slot start.
                def _dist(row):
                    try:
                        st = parse_iso_utc(row["startTime"]).astimezone(tz)
                        return abs((st - active_start_local).total_seconds())
                    except Exception:
                        return 10**9

                trip_sorted = sorted(trip, key=lambda r: (rank_type(r.get("type","")), _dist(r)))
                chosen = trip_sorted[0] if trip_sorted else None

            if chosen:
                se = local_start_end(chosen)
                chosen_buy = pick_buy(chosen)
                chosen_sell = pick_sell(chosen)

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

                    pending_override = (target_label, chosen_buy, chosen_sell)

                    # Human-friendly summary (with advanced→perKwh fallback for BUY)
                    labels, buy_vals, sell_vals = [], [], []
                    for r in trip:
                        try:
                            st = parse_iso_utc(r["startTime"]).astimezone(tz)
                            en = parse_iso_utc(r["endTime"]).astimezone(tz)
                            lbl2 = _label_from_range(st, en)  # normalized
                        except Exception:
                            lbl2 = "<unknown>"

                        # BUY: if --advanced-price use advancedPrice.<field> with fallback to perKwh
                        if args.advanced_price:
                            adv = (r.get("advancedPrice") or {})
                            b = adv.get(args.advanced_price)
                            if b is None:
                                b = r.get("perKwh")
                        else:
                            b = r.get("perKwh")

                        # SELL
                        s = r.get("spotPerKwh")

                        labels.append(lbl2)
                        buy_vals.append(None if b is None else round(float(b), 2))
                        sell_vals.append(None if s is None else round(float(s), 2))

                    print("[amber2sigen] Current window BUY slots = " +
                          ", ".join(f"{l}:{('<nil>' if b is None else b)}" for l, b in zip(labels, buy_vals)),
                          file=sys.stderr)
                    print("[amber2sigen] Current window SELL slots = " +
                          ", ".join(f"{l}:{('<nil>' if s is None else s)}" for l, s in zip(labels, sell_vals)),
                          file=sys.stderr)
                else:
                    print("[amber2sigen] /current row lacked parseable start/end; skipping override.", file=sys.stderr)
            else:
                print("[amber2sigen] No suitable /current row found for active slot override.", file=sys.stderr)

    # ---- Rotate + canonicalize to midnight day labels BEFORE applying override ----
    buy_ranges = rotate_series_to_midnight(buy_ranges)
    sell_ranges = rotate_series_to_midnight(sell_ranges)
    buy_ranges = canonicalize_series_to_day(buy_ranges, step_min)
    sell_ranges = canonicalize_series_to_day(sell_ranges, step_min)

    # ---- Apply pending override by label with normalization + index fallback (v24) ----
    if pending_override:
        target_label, chosen_buy, chosen_sell = pending_override

        # --- BUY ---
        # Attempt exact match with normalization
        buy_idx = next((i for i, (tr, _) in enumerate(buy_ranges)
                        if _norm_label(tr) == _norm_label(target_label) or
                           _norm_label(tr).split("-", 1)[0] == _norm_label(target_label).split("-", 1)[0]), None)
        if buy_idx is None:
            print(f"[amber2sigen] Could not locate BUY target label {target_label} in final series.", file=sys.stderr)
            buy_idx = _half_hour_index_from_label(target_label) if step_min == 30 else None
            if buy_idx is None or not (0 <= buy_idx < len(buy_ranges)):
                print(f"[amber2sigen] FATAL: cannot compute index for BUY label {target_label}; no override applied.", file=sys.stderr)
            else:
                idx_label, idx_old = buy_ranges[buy_idx]
                if chosen_buy is None or chosen_buy == 0.0:
                    ZERO_EVENTS_BUY.append(f"current {idx_label}")
                    print(f"[amber2sigen] Skipping BUY current override (zero/missing) for {idx_label}; keeping forecast {idx_old}", file=sys.stderr)
                else:
                    buy_new = round(float(chosen_buy), 2)
                    buy_ranges[buy_idx] = (idx_label, buy_new)
                    print(f"[amber2sigen] Overrode {idx_label} BUY: {idx_old:.2f} → {buy_new:.2f} (from 5-min CurrentInterval)", file=sys.stderr)
        else:
            old = buy_ranges[buy_idx][1]
            if chosen_buy is None or chosen_buy == 0.0:
                ZERO_EVENTS_BUY.append(f"current {buy_ranges[buy_idx][0]}")
                print(f"[amber2sigen] Skipping BUY current override (zero/missing) for {buy_ranges[buy_idx][0]}; keeping forecast {old}", file=sys.stderr)
            else:
                buy_new = round(float(chosen_buy), 2)
                lbl = buy_ranges[buy_idx][0]
                buy_ranges[buy_idx] = (lbl, buy_new)
                print(f"[amber2sigen] Overrode {lbl} BUY: {old:.2f} → {buy_new:.2f} (from 5-min CurrentInterval)", file=sys.stderr)

        # --- SELL ---
        sell_idx = next((i for i, (tr, _) in enumerate(sell_ranges)
                         if _norm_label(tr) == _norm_label(target_label) or
                            _norm_label(tr).split("-", 1)[0] == _norm_label(target_label).split("-", 1)[0]), None)
        if sell_idx is None:
            print(f"[amber2sigen] Could not locate SELL target label {target_label} in final series.", file=sys.stderr)
            sell_idx = _half_hour_index_from_label(target_label) if step_min == 30 else None
            if sell_idx is None or not (0 <= sell_idx < len(sell_ranges)):
                print(f"[amber2sigen] FATAL: cannot compute index for SELL label {target_label}; no override applied.", file=sys.stderr)
            else:
                idx_label, idx_old = sell_ranges[sell_idx]
                if chosen_sell is None:
                    print(f"[amber2sigen] Skipping SELL current override (missing) for {idx_label}; keeping forecast {idx_old}", file=sys.stderr)
                else:
                    sell_new = round(float(chosen_sell), 2)
                    sell_ranges[sell_idx] = (idx_label, sell_new)
                    print(f"[amber2sigen] Overrode {idx_label} SELL: {idx_old:.2f} → {sell_new:.2f} (from 5-min CurrentInterval)", file=sys.stderr)
        else:
            old = sell_ranges[sell_idx][1]
            if chosen_sell is None:
                print(f"[amber2sigen] Skipping SELL current override (missing) for {sell_ranges[sell_idx][0]}; keeping forecast {old}", file=sys.stderr)
            else:
                sell_new = round(float(chosen_sell), 2)
                lbl = sell_ranges[sell_idx][0]
                sell_ranges[sell_idx] = (lbl, sell_new)
                print(f"[amber2sigen] Overrode {lbl} SELL: {old:.2f} → {sell_new:.2f} (from 5-min CurrentInterval)", file=sys.stderr)

    # Postbuild zero scan (final series)
    for tr, p in buy_ranges:
        if p == 0.0:
            ZERO_EVENTS_BUY.append(f"postbuild {tr}")

    # End-of-run diagnostics: active slot BUY/SELL in final series (normalized lookup)
    print(f"[amber2sigen] Active slot BUY {current_label} = "
          f"{_lookup_price_by_label(buy_ranges, current_label) or '<not found>'}", file=sys.stderr)
    print(f"[amber2sigen] Active slot SELL {current_label} = "
          f"{_lookup_price_by_label(sell_ranges, current_label) or '<not found>'}", file=sys.stderr)

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
