FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY amber_to_sigen.py .
COPY sigen_make_env.py .

# Create directory for config
RUN mkdir -p /config

# Set timezone support
ENV TZ=Australia/Sydney

# Run the script with environment variables
CMD ["python", "-u", "amber_to_sigen.py"]
