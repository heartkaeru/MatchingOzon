# MatchingOzon

## Docker

Сборка образа:

```bash
docker build -t matching-ozon:latest .
```

Запуск инференса:

```bash
docker run --rm \
  -v "$PWD/data:/data" \
  matching-ozon:latest \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output-path /data/submit.csv
```

Образ запускает `submission/run.py`. Если в `submission/weights/` лежат файлы `catboost_fold_*.cbm` или `catboost_model.cbm`, они попадут в образ при сборке.
