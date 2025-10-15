#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f ./amber2sigen.env ]]; then
  echo "[run.sh] Missing /etc/amber2sigen.env"; exit 1
fi
source /etc/amber2sigen.env

: "${AMBER_TOKEN:?}"
: "${STATION_ID:?}"
: "${SIGEN_USER:?}"
: "${SIGEN_DEVICE_ID:?}"
: "${SIGEN_PASS_ENC:?}"

PY=python3
ARGS=(
  "--station-id" "${STATION_ID}"
  "--amber-token" "${AMBER_TOKEN}"
  "--interval" "${INTERVAL:-30}"
  "--tz" "${TZ_OVERRIDE:-Australia/Adelaide}"
  "--align" "${ALIGN:-end}"
  "--plan-name" "${PLAN_NAME:-Amber Live}"
  "--advanced-price" "${ADVANCED-predicted}"
  "--sigen-user" "${SIGEN_USER}"
  "--device-id" "${SIGEN_DEVICE_ID}"
)

# Handle USE_CURRENT env var (0/1 or False/True)
case "${USE_CURRENT,,}" in
  1|true|yes|on)
    ARGS+=(--use-current)
    ;;
  0|false|no|off)
    ARGS+=(--no-use-current)
    ;;
  *)
    # default: let Python handle its own default
    ;;
esac

[[ "${1:-}" == "--dry-run" ]] && ARGS+=("--dry-run")

echo "[run.sh] Exec: ${PY} amber_to_sigen.py ${ARGS[*]}"
exec "${PY}" amber_to_sigen.py "${ARGS[@]}"
