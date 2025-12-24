#!/usr/bin/env bash
set -euo pipefail

# ===== Config (env -> flags) =====
: "${TZ_OVERRIDE:=${TZ:-Australia/Adelaide}}"
: "${ALIGN:=end}"
: "${PLAN_NAME:=Amber Live}"
: "${ADVANCED:=predicted}"
: "${USE_CURRENT:=1}"
: "${RUN_IMMEDIATELY:=0}"     # set to 1 to run once on start before aligning
: "${EXTRA_ARGS:=}"           # optional passthrough flags
: "${INTERVAL:=30}"           # 5 or 30 minute intervals
: "${SELL_CHANNEL:=general}"  # general or feedIn
: "${SELLADVANCED:=}"         # low, predicted, high (for feedIn only)

# Required secrets/IDs
: "${AMBER_TOKEN:?AMBER_TOKEN missing}"
: "${STATION_ID:?STATION_ID missing}"
: "${SIGEN_USER:?SIGEN_USER missing}"
: "${SIGEN_PASS_ENC:?SIGEN_PASS_ENC missing}"
: "${SIGEN_DEVICE_ID:?SIGEN_DEVICE_ID missing}"

# Clean exit on stop
quit() { echo "[amber2sigen] received signal, exiting"; exit 0; }
trap quit TERM INT

# Compute sleep until the next */5 minute boundary at +30s.
# Uses epoch math (timezone-agnostic; 5-min boundaries don't drift on DST).
sleep_to_next_tick() {
  local now next sleep_s
  now=$(date +%s)
  # Next 5-min boundary (multiples of 300s) + 30s
  next=$(( (now/300)*300 + 300 + 30 ))
  sleep_s=$(( next - now ))
  # If we started right after :30, next would be negative/zero -> add one more 5-min window
  if [ "$sleep_s" -le 0 ]; then
    sleep_s=$(( sleep_s + 300 ))
  fi
  echo "[amber2sigen] sleeping ${sleep_s}s to next 5-min +30s tick (TZ=${TZ_OVERRIDE})" >&2
  sleep "${sleep_s}"
}

run_once() {
  set +e

  # Build command args
  local cmd_args=(
    --amber-token "${AMBER_TOKEN}"
    --station-id "${STATION_ID}"
    --tz "${TZ_OVERRIDE}"
    --align "${ALIGN}"
    --plan-name "${PLAN_NAME}"
    --sigen-user "${SIGEN_USER}"
    --device-id "${SIGEN_DEVICE_ID}"
    --interval "${INTERVAL}"
    --sell-channel "${SELL_CHANNEL}"
  )

  # Add advanced-price if set
  if [ -n "${ADVANCED:-}" ]; then
    cmd_args+=(--advanced-price "${ADVANCED}")
  fi

  # Add sell-advanced-price if set
  if [ -n "${SELLADVANCED:-}" ]; then
    cmd_args+=(--sell-advanced-price "${SELLADVANCED}")
  fi

  # Add use-current flag if enabled
  if [ "${USE_CURRENT,,}" = "1" ] || [ "${USE_CURRENT,,}" = "true" ]; then
    cmd_args+=(--use-current)
  fi

  # Run the script
  python -u amber_to_sigen.py "${cmd_args[@]}" ${EXTRA_ARGS}
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    date +%s > /tmp/amber2sigen.lastok
  else
    echo "[amber2sigen] run failed with exit ${rc}" >&2
  fi
  return "$rc"
}

echo "[amber2sigen] start (self-aligning loop; target: */5 @ :30)" >&2
echo "[amber2sigen] plan='${PLAN_NAME}' tz='${TZ_OVERRIDE}' interval=${INTERVAL}" >&2
echo "[amber2sigen] advanced='${ADVANCED:-off}' sell_channel='${SELL_CHANNEL}' sell_advanced='${SELLADVANCED:-off}'" >&2
echo "[amber2sigen] use_current=${USE_CURRENT} extra='${EXTRA_ARGS}'" >&2

# Optional immediate first run, then align thereafter
if [ "${RUN_IMMEDIATELY}" = "1" ] || [ "${RUN_IMMEDIATELY,,}" = "true" ]; then
  echo "[amber2sigen] RUN_IMMEDIATELY=1 -> running once before first alignment" >&2
  run_once || true
fi

# Main loop: align -> run -> repeat
while true; do
  sleep_to_next_tick
  run_once || true
done
