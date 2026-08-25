FROM odsai/ecup26-matching-baseline:1.0

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    catboost==1.2.10 \
    rapidfuzz==3.14.5 \
    lightgbm==4.7.0 \
    nltk==3.10.3
