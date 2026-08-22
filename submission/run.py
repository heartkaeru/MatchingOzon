"""
Inference entry point script for MatchingOzon (E-CUP 2026, Задача 1).

CLI (точное соответствие спецификации соревнования — не менять на дефисы):
  --items_path    путь к items.parquet
  --matches_path  путь к matches.parquet (пары id1/id2-кандидатов)
  --output-path   путь для submit.csv (id1,id2,predict)

Полностью автономный, read-only, offline пайплайн:
чтение -> векторизация -> генерация признаков -> предикт CatBoost -> submit.csv

Ограничения проверки: Check 1 мин / Public (~115k пар) 6 мин / Private (~275k пар) 13 мин.
Ресурсы: 20 CPU, 200GB RAM, NVidia H100 80GB. Никакого доступа в интернет.

Схема данных:
  items.parquet:   id, name, attributes (JSON-строка), category
  matches.parquet: id1, id2  (на инференсе БЕЗ target — target есть только
                    в обучающих matches.parquet/matches_llm.parquet)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

# ВАЖНО (упаковка сабмита): в архив соревнования уходит только содержимое
# папки submission/ (metadata.json, run.py, weights/) — src/ лежит вне
# submission/ в репозитории и сам по себе в архив НЕ попадёт.
# make_submission.py должен копировать src/ ВНУТРЬ submission/ (т.е. в
# архиве run.py и src/ оказываются рядом, в одной директории) —
# тогда этот sys.path.insert (директория самого run.py) найдёт src
# и на закрытом стенде, и локально при запуске из submission/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.encoder import CatalogEncoder
from src.feature_builder import build_pair_features

WEIGHTS_DIR = Path(__file__).parent / "weights"


def parse_args():
    parser = argparse.ArgumentParser(description="Inference CLI for MatchingOzon")
    parser.add_argument(
        "--items_path",
        required=True,
        help="Path to items catalog (parquet), e.g. items.parquet",
    )
    parser.add_argument(
        "--matches_path",
        required=True,
        help="Path to candidate pairs (parquet), e.g. matches.parquet",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write predictions, e.g. submit.csv",
    )
    return parser.parse_args()


def load_items(items_path):
    return pd.read_parquet(items_path)


def load_matches(matches_path):
    return pd.read_parquet(matches_path)


def load_encoder():
    # CatalogEncoder(model_path=...) грузит веса сам в __init__ —
    # путь передаём явно, а не полагаемся на дефолт (на закрытом стенде
    # интернета нет, так что model_path обязателен, а не None -> HF hub).
    return CatalogEncoder(model_path=WEIGHTS_DIR / "encoder_fp16.onnx")


def load_classifier():
    classifier = CatBoostClassifier()
    classifier.load_model(str(WEIGHTS_DIR / "catboost_model.cbm"))
    return classifier


def build_item_text(row):
    """
    Собирает текст для энкодера из name + category + распакованных attributes.

    ПРЕДПОЛОЖЕНИЕ (сверить с Role 2 / NLP engineer): в описании их роли
    сказано "объединение заголовков, брендов и атрибутов в информативный
    текст", но отдельного поля brand в items.parquet нет — судя по
    глоссарию, только id/name/attributes/category, и бренд, скорее всего,
    зашит внутри attributes. Если Role 2 дообучала энкодер на другом
    склеенном формате текста (другой порядок полей, другие разделители,
    без category и т.п.) — эмбеддинги будут мимо распределения, на котором
    учился энкодер, и это никак не проявится как ошибка, только как
    просевшее качество. Нужно свериться перед тем как это едет в сабмит.
    """
    parts = [str(row["name"])]
    category = row.get("category")
    if pd.notna(category):
        parts.append(str(category))

    attrs_raw = row.get("attributes")
    if isinstance(attrs_raw, str) and attrs_raw.strip():
        try:
            attrs = json.loads(attrs_raw)
            if isinstance(attrs, dict):
                parts.extend(f"{k}: {v}" for k, v in attrs.items())
            else:
                parts.append(str(attrs))
        except (TypeError, ValueError):
            parts.append(attrs_raw)

    return " ".join(parts)


def build_embeddings(encoder, items):
    """
    Прогоняет CatalogEncoder.encode() по каталогу и возвращает
    {id: embedding} для последующего склеивания с товарами в парах.
    """
    item_ids = items["id"].tolist()
    texts = items.apply(build_item_text, axis=1).tolist()
    vectors = encoder.encode(texts)
    return dict(zip(item_ids, vectors))


def build_features(items, matches, embeddings):
    """
    Собирает парный DataFrame с суффиксами _1/_2 и одним батчевым вызовом
    build_pair_features() получает матрицу фичей (см. src/feature_builder.py).

    matches на инференсе содержит только id1/id2 (без target — target есть
    только в обучающих matches.parquet/matches_llm.parquet).

    ПРИМЕЧАНИЕ: в items.parquet нет полей brand/price (бренд зашит в
    attributes JSON), поэтому brand_match вернёт 0, а price_diff/price_ratio
    — NaN, пока не появится парсер attributes. Эмбеддинги сейчас фичами
    не используются, но расчёт оставлен для следующих итераций.
    """
    items_indexed = items.set_index("id")

    feature_cols = [c for c in ("name", "brand", "category", "price") if c in items.columns]
    left = items_indexed.loc[matches["id1"], feature_cols].reset_index(drop=True)
    right = items_indexed.loc[matches["id2"], feature_cols].reset_index(drop=True)

    pairs = pd.concat([left.add_suffix("_1"), right.add_suffix("_2")], axis=1)
    return build_pair_features(pairs)


def predict(classifier, features):
    """
    Возвращает непрерывный скор (вероятность класса "дубликат"), НЕ 0/1 —
    метрика соревнования Macro PR-AUC (average_precision_score) требует
    ранжирующего скора, а не бинарной метки.

    ВАЖНО: порядок/состав колонок features должен точно совпадать с тем,
    на чём обучался classifier (train/03_train_catboost.py) — если
    build_pair_features() в src/feature_builder.py вернёт колонки в другом
    порядке или с другими именами, CatBoost либо упадёт, либо (что хуже)
    тихо предскажет по неверно сопоставленным фичам. Свериться с Role 1.
    """
    return classifier.predict_proba(features)[:, 1]


def write_submission(matches, predictions, output_path):
    submission = pd.DataFrame(
        {
            "id1": matches["id1"].values,
            "id2": matches["id2"].values,
            "predict": predictions,
        }
    )
    submission.to_csv(output_path, index=False)


def main():
    start = time.perf_counter()
    args = parse_args()
    print(f"[info] items_path={args.items_path} matches_path={args.matches_path} output_path={args.output_path}")

    items = load_items(args.items_path)
    matches = load_matches(args.matches_path)
    print(f"[info] loaded items={len(items)} matches={len(matches)} ({time.perf_counter() - start:.1f}s)")

    encoder = load_encoder()
    embeddings = build_embeddings(encoder, items)
    print(f"[info] embeddings ready ({time.perf_counter() - start:.1f}s)")

    classifier = load_classifier()
    features = build_features(items, matches, embeddings)
    print(f"[info] features ready ({time.perf_counter() - start:.1f}s)")

    predictions = predict(classifier, features)
    write_submission(matches, predictions, args.output_path)
    print(f"[info] done, total {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
