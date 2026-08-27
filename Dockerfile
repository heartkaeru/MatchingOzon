FROM odsai/ecup26-matching-baseline:1.0

USER root

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN python -m pip install --no-cache-dir --no-deps catboost rapidfuzz six || \
    python3 -m pip install --no-cache-dir --no-deps catboost rapidfuzz six

WORKDIR /app
COPY . /app

WORKDIR /app/submission

ENTRYPOINT ["python", "-u", "run.py"]

