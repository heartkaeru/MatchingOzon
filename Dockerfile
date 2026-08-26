FROM odsai/ecup26-matching-baseline:1.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src ./src
COPY submission ./submission
COPY train ./train
COPY mlops ./mlops
COPY make_submission.py README.md ./

WORKDIR /app/submission

ENTRYPOINT ["python", "-u", "run.py"]

