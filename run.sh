#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-/etc/amber2sigen.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[run.sh] Missing ${ENV_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

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
  "--advanced-price" "${ADVANCED:-predicted}"
  "--sigen-user" "${SIGEN_USER}"
  "--device-id" "${SIGEN_DEVICE_ID}"
)

# v29: SELL channel selection (general|feedIn) -> python expects lowercased choices (general|feedin)
SELL_CH="${SELL_CHANNEL:-general}"
SELL_CH_LC="$(echo "${SELL_CH}" | tr '[:upper:]' '[:lower:]')"
ARGS+=( "--sell-channel" "${SELL_CH_LC}" )

# v29: SELL advanced price selection for feed-in (low|predicted|high)
if [[ -n "${SELLADVANCED:-}" ]]; then
  ARGS+=( "--sell-advanced-price" "${SELLADVANCED}" )
fi

# Optional: slot shift
if [[ -n "${SLOT_SHIFT:-}" ]]; then
  ARGS+=( "--slot-shift" "${SLOT_SHIFT}" )
fi

# Optional: allow-zero-buy (guardrail override)
case "${ALLOW_ZERO_BUY:-0}" in
  1|true|yes|on|TRUE|YES|ON) ARGS+=( "--allow-zero-buy" ) ;;
esac

# Handle USE_CURRENT env var (0/1 or False/True)
case "${USE_CURRENT:-1,,}" in
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
