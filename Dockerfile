FROM odsai/ecup26-matching-baseline:1.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --prefer-binary catboost rapidfuzz

WORKDIR /app
COPY . /app

WORKDIR /app/submission

ENTRYPOINT ["python", "-u", "run.py"]

