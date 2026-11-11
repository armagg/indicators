FROM python:3.15-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pytest is included on purpose: the image is also how research jobs and the
# suite run on the servers, and a deployed artifact you cannot test on the
# box it runs on is one you are trusting blind.
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY app.py conftest.py ./
COPY ta/ ./ta/
COPY tests/ ./tests/
COPY research/ ./research/

# Research runs write here; mount a volume to keep results across containers.
RUN mkdir -p /app/cache
VOLUME ["/app/cache"]

EXPOSE 5000

# Default is the HTTP service. Research jobs override the command, e.g.
#   docker run --rm -v ta_cache:/app/cache indicators \
#       python -m research.ma_inventory_brute cache/usdtirt_1H.csv --out cache
CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
