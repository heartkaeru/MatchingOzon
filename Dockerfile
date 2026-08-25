FROM odsai/ecup26-matching-baseline:1.0

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir catboost rapidfuzz

