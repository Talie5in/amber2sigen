# Amber → Sigen Energy Controller

This project syncs real-time Amber Electric import/export prices into your Sigen Energy Controller.  
It can run every 5 or 30 minutes via systemd timer to keep your Sigen tariffs up to date.

---

## Features

- Pulls **Amber API** 5-minute or 30-minute prices (`perKwh` for buy, `spotPerKwh` for sell).
- Supports `--advanced-price {low,predicted,high}` for Amber’s `advancedPrice`.
- Seeds the **current active slot** from Amber’s `/prices/current` (optional, enabled by default).
- Authenticates to **Sigen Cloud** with either:
  - `SIGEN_USER`, `SIGEN_PASS_ENC`, `SIGEN_DEVICE_ID` (recommended), or
  - `SIGEN_BEARER` (manual bearer token).
- Dry-run mode prints JSON payload without posting to Sigen.
- Works with both `--interval 5` and `--interval 30`.
  - **Note**: Sigen only supports `--interval 30` right now, setting to 5 will break writing to Sigen.
- Supports 30 minute billing for Amber customers (Eg, Victoria?)
  - Use `--no-use-current` or `USE_CURRENT=False` ENV Flag - to force 30 minute Amber billing data (aka "Current Active Slot")

---

## Installation Options

Choose one of the following installation methods:

- **[Docker](#docker-installation)** (Recommended) - Easiest setup, automatic restarts
- **[Manual/systemd](#manual-installation)** - Traditional Linux service installation

---

## Docker Installation

### Prerequisites

- Docker and Docker Compose installed
- Amber API token from https://app.amber.com.au/developers
- Sigen credentials (see [How to find SIGEN_PASS_ENC and SIGEN_DEVICE_ID](#how-to-find-sigen_pass_enc-and-sigen_device_id))

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/bolagnaise/amber2sigen.git
   cd amber2sigen
   ```

2. **Configure environment**

   Copy the example env file and edit with your credentials:

   ```bash
   cp amber2sigen.env.example amber2sigen.env
   nano amber2sigen.env
   ```

   ```dotenv
   AMBER_TOKEN=psk_xxxxxxxxxxxxxxxxxxxx
   SIGEN_USER=your@email.com
   SIGEN_DEVICE_ID=1756353655250
   SIGEN_PASS_ENC="ENCRYPTED_BLOB"

   # Optional tuning
   INTERVAL=30
   TZ_OVERRIDE=Australia/Adelaide
   ALIGN=end
   PLAN_NAME=Amber Live
   ADVANCED=predicted
   USE_CURRENT=1
   STATION_ID=<Your Station ID>
   ```

3. **Start the container**

   ```bash
   docker compose up -d
   ```

4. **View logs**

   ```bash
   docker compose logs -f
   ```

### Docker Commands

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Restart
docker compose restart

# View logs
docker compose logs -f amber2sigen

# Rebuild after updates
docker compose build --no-cache
docker compose up -d
```

---

## Manual Installation

### Requirements

- Python 3.9+
- `requests`, `pycryptodome`, `python-dateutil`

Install:

```bash
pip install -r requirements.txt
```

---

### Step 1. Create user and directory

For security, run this under a dedicated system user:

```bash
sudo useradd --system --home /opt/amber2sigen --shell /usr/sbin/nologin amber2sigen
sudo mkdir -p /opt/amber2sigen
sudo chown amber2sigen:amber2sigen /opt/amber2sigen
```

Clone or copy this repo into `/opt/amber2sigen`.
Ensure the python script and run.sh files are also owned by amber2sigen:amber2sigen

---

## Step 2. Generate `.env` file

Run the helper to create your env file:

```bash
cd /opt/amber2sigen
python3 sigen_make_env.py
```

You will be prompted for:

- **Amber API token** (`AMBER_TOKEN`)
- **Sigen username** (`SIGEN_USER`) (See Below)
- **Sigen encoded password** (`SIGEN_PASS_ENC`) (See below)
- **Sigen device ID** (`SIGEN_DEVICE_ID`) (See below)

Then move the generated env file to `/etc`:

```bash
sudo mv amber2sigen.env /etc/amber2sigen.env
sudo chown root:root /etc/amber2sigen.env
sudo chmod 600 /etc/amber2sigen.env
```

### How to find SIGEN_PASS_ENC and SIGEN_DEVICE_ID

1. Open the Sigen web portal in your browser (https://app-aus.sigencloud.com/)  
2. Open Developer Tools → **Network** tab.  
3. Log in normally.  
4. Look for a request to:  

   ```text
   https://api-aus.sigencloud.com/auth/oauth/token
   ```

5. In the request payload you will see:
   - `password` → this is the **encoded password** (copy into `SIGEN_PASS_ENC`).  
   - `userDeviceId` → this is the **device ID** (copy into `SIGEN_DEVICE_ID`).  

Copy these values exactly into the prompts.

## How to find your Sigen Station ID

The `STATION_ID` is a unique numeric ID assigned to your Sigen Energy Controller by Sigen Cloud.  
It must be included in the payload or Sigen won’t know which unit to update.

Easiest way to find it:

1. Ask SigenAI to "Tell me my StationID"

Complex Ways to find it:

1. **HAR capture**: In your browser, open the Sigen web portal, perform a tariff save, then export the HAR.  
   Look for `"stationId": <your station ID>` in the JSON payload.  
2. **App/device info**: Sometimes shown in the app under device details.  
3. Once known, add it to your `.env` file as `STATION_ID=...`.

This value is specific to your unit — not random or generated locally.

---

## Step 3. Example `/etc/amber2sigen.env`

```dotenv
AMBER_TOKEN=psk_xxxxxxxxxxxxxxxxxxxx
SIGEN_USER=your@email.com
SIGEN_DEVICE_ID=1756353655250
SIGEN_PASS_ENC="ENCRYPTED_BLOB"

# Optional tuning
INTERVAL=30
TZ_OVERRIDE=Australia/Adelaide
ALIGN=end
PLAN_NAME=Amber Live
ADVANCED=predicted
USE_CURRENT=True
STATION_ID=<Ask SigenAI for your Station ID>
PAYLOAD_DEBUG=1
```

---

## Step 4. Run manually

```bash
cd /opt/amber2sigen
sudo -u amber2sigen bash run.sh --dry-run
sudo -u amber2sigen bash run.sh
```

---

## Step 5. systemd

`/etc/systemd/system/amber2sigen.service`

```ini
[Unit]
Description=Amber -> Sigen price sync
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/etc/amber2sigen.env
WorkingDirectory=/opt/amber2sigen
ExecStart=/bin/bash /opt/amber2sigen/run.sh
User=amber2sigen
Group=amber2sigen
```

`/etc/systemd/system/amber2sigen.timer`

```ini
[Unit]
Description=Run amber2sigen periodically at absolute 5-min marks +20s
Wants=amber2sigen.service

[Timer]
OnBootSec=2min
OnCalendar=*:0/5:20
Unit=amber2sigen.service
Persistent=true
AccuracySec=1s

[Install]
WantedBy=timers.target
```

Enable:

```bash
sudo systemctl daemon-reexec
sudo systemctl enable --now amber2sigen.timer
```

---

## Journalctl Troubleshooting

- Check last run:

  ```bash
  journalctl -u amber2sigen.service -n 200 --no-pager
  ```

- Follow logs:

  ```bash
  journalctl -u amber2sigen.service -f
  ```

- Verify timer:

  ```bash
  systemctl list-timers | grep amber2sigen
  ```

- Watch Timer:

  ```bash
  watch -n 10 "systemctl list-timers | grep amber2sigen"
  ```

---

# CLI Flags

## CLI Flags for `amber_to_sigen.py`

| Flag | Type / Values | Default | Purpose |
|------|---------------|---------|---------|
| `--amber-token` | String | from env `AMBER_TOKEN` | Amber API token used to authenticate and fetch prices. |
| `--site-id` | String | auto-detected | Amber site ID. If omitted, first active site is used. |
| `--tz` | Timezone string | `Australia/Adelaide` | Time zone for converting and labeling price slots. |
| `--interval` | `5` or `30` | from env `INTERVAL` (default `30`) | Slot size in minutes; determines whether 5-minute or 30-minute Amber prices are used. |
| `--align` | `start` / `end` | `end` | Whether to align Amber rows to slot **start** or **end**. Amber app convention is `end`. |
| `--slot-shift` | Integer | `0` | Rotate/shift the entire BUY/SELL series by N slots. Positive = later, negative = earlier. |
| `--advanced-price` | `low` / `predicted` / `high` | none | Use Amber’s `advancedPrice.<field>` for BUY price instead of `perKwh`. |
| `--use-current` | Flag (on/off) | true (unless `USE_CURRENT=False`) | Enable fetching `/prices/current` to override the current slot with the latest interval. |
| `--no-use-current` | Flag (on/off) | — | Explicitly disable `/prices/current` override. |
| `--station-id` | Integer | **Required** | The numeric `stationId` for your Sigen Energy Controller. |
| `--plan-name` | String | `"SAPN TOU"` | Label to use for the tariff plan in the payload sent to Sigen. |
| `--sigen-url` | URL | `SIGEN_SAVE_URL` or default Sigen API URL | Endpoint to POST prices into Sigen Cloud. |
| `--sigen-user` | String | from env `SIGEN_USER` | Sigen account username (email). Used for authentication. |
| `--device-id` | String | from env `SIGEN_DEVICE_ID` or `"1756353655250"` | Sigen `userDeviceId`. Required for password-grant authentication. |
| `--dry-run` | Flag | off | Print JSON payload instead of POSTing to Sigen. Useful for testing. |
| `--allow-zero-buy` | Flag | off | Allow posting to Sigen even if final BUY prices include `0.0`. By default, posting is skipped if zeros are detected. |

### CLI Notes for `amber_to_sigen.py`

- Environment variables (`AMBER_TOKEN`, `INTERVAL`, `SIGEN_*`, etc.) can be set in `/etc/amber2sigen.env` instead of CLI flags.  
- By default, if BUY prices contain `0.0` the script **will not POST** unless `--allow-zero-buy` is specified.  
- `/prices/current` override improves accuracy by seeding with the **current active slot**. Disable with `--no-use-current`.

---

## CLI Flags for `sigen_make_env.py`

| Flag | Type / Values | Default | Purpose |
|------|---------------|---------|---------|
| `--user` | String | **Required** | Sigen account email/username. |
| `--password` | String | **Required** | (Old behavior) Plaintext Sigen password to be AES-encoded. ⚠️ In the new version, you should instead supply the encoded password via prompt. |
| `--device-id` | String (13-digit) | Randomly generated if omitted | Sigen `userDeviceId`. If not supplied, a random 13-digit ID is generated. |
| `--env-path` | Path | `amber2sigen.env` | Output path for the generated environment file. |
| `--overwrite` | Flag | off | Allow overwriting an existing env file. |
| `--interval` | `5` or `30` | `30` | Default Amber interval in minutes. |
| `--tz` | Timezone string | `Australia/Adelaide` | Default time zone override. |
| `--align` | `start` / `end` | `end` | Default slot alignment label. |
| `--plan-name` | String | `Amber Live` | Default plan name label. |
| `--advanced` | `low` / `predicted` / `high` | `predicted` | Default advanced price field to use for BUY. |
| `--use-current` | `0` or `1` | `1` | Whether to use Amber’s `/prices/current` endpoint for active slot seeding. |

### CLI Notes for `sigen_make_env.py`

- The script writes a clean `.env` file with the variables needed by `amber_to_sigen.py` and `run.sh`.  
- In the newer workflow, `--password` encoding must be bypassed (not used) and replaced with manual entry of `SIGEN_PASS_ENC` from browser dev tools.  