FROM python:3.11-slim

# tzdata for local time printing (alignment is epoch-based and TZ-safe)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tzdata ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -u 10001 -m appuser

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY amber_to_sigen.py .
COPY sigen_make_env.py .

# Copy entrypoint script
COPY docker/run.sh /app/docker/run.sh
RUN chmod +x /app/docker/run.sh

# Create directory for config
RUN mkdir -p /config && chown appuser:appuser /config

ENV TZ=Australia/Adelaide \
    PYTHONUNBUFFERED=1

# Health: OK if we've had a successful run in last 10 minutes
HEALTHCHECK --interval=2m --timeout=10s CMD bash -lc '[ -f /tmp/amber2sigen.lastok ] && find /tmp/amber2sigen.lastok -mmin -10 | grep -q .'

USER appuser
ENTRYPOINT ["/app/docker/run.sh"]
