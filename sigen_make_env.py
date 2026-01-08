#!/usr/bin/env python3
# sigen_make_env.py
# Prompt for Amber/Sigen credentials and create amber2sigen.env (v29)

import getpass
from pathlib import Path
import os

def write_env(path: Path, values: dict, overwrite: bool):
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path}\nPass --overwrite to replace it.")
    lines = []
    for k, v in values.items():
        v = (v or "").replace("\n", "")
        if any(ch in v for ch in [' ', '#', '"', "'", '$']):
            v_out = f'"{v}"'
        else:
            v_out = v
        lines.append(f"{k}={v_out}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def main():
    print("Amber → Sigen Environment Setup (v29)")

    amber_token = getpass.getpass("Enter Amber API Token (AMBER_TOKEN): ")
    station_id = input("Enter Sigen station ID (STATION_ID): ").strip()

    sigen_user = input("Enter Sigen username/email (SIGEN_USER): ").strip()
    sigen_pass = getpass.getpass("Enter Sigen password (SIGEN_PASS): ")
    sigen_device_id = input("Enter Sigen device ID (SIGEN_DEVICE_ID): ").strip()

    # v29: SELL channel + optional advanced selector
    sell_channel = input("SELL channel (SELL_CHANNEL) [general|feedIn] (default: general): ").strip() or "general"
    sell_advanced = input("SELL advanced price (SELLADVANCED) [low|predicted|high] (blank=off): ").strip()

    # Optional knobs
    slot_shift = input("Slot shift (SLOT_SHIFT) integer (default: 0): ").strip() or "0"
    allow_zero_buy = input("Allow zero BUY prices? (ALLOW_ZERO_BUY) [0/1] (default: 0): ").strip() or "0"
    payload_debug = input("Print payload JSON? (PAYLOAD_DEBUG) [0/1] (default: 1): ").strip() or "1"

    values = {
        "AMBER_TOKEN": amber_token,
        "STATION_ID": station_id,
        "SIGEN_USER": sigen_user,
        "SIGEN_PASS": sigen_pass,
        "SIGEN_DEVICE_ID": sigen_device_id,

        "INTERVAL": "30",
        "TZ_OVERRIDE": "Australia/Adelaide",
        "ALIGN": "end",
        "PLAN_NAME": "Amber Live",
        "ADVANCED": "predicted",
        "USE_CURRENT": "1",

        # v29 additions
        "SELL_CHANNEL": sell_channel,
        "SELLADVANCED": sell_advanced,

        # Optional
        "SLOT_SHIFT": slot_shift,
        "ALLOW_ZERO_BUY": allow_zero_buy,
        "PAYLOAD_DEBUG": payload_debug,
    }

    env_path = Path("amber2sigen.env")
    write_env(env_path, values, overwrite=True)
    print(f"Wrote {env_path.resolve()}")

if __name__ == "__main__":
    main()
