"""
Скрипт точки входа для инференса MatchingOzon (E-CUP 2026, Задача 1).

CLI (точное соответствие спецификации соревнования — имена аргументов не менять):
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

import numpy as np
import onnxruntime as ort
import pandas as pd

# ВАЖНО (упаковка сабмита): в архив соревнования уходит только содержимое
# Поиск модулей src как в submission/, так и в корне проекта
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_builder import build_pair_features

WEIGHTS_DIR = Path(__file__).parent / "weights"


def parse_args():
    parser = argparse.ArgumentParser(description="CLI инференса для MatchingOzon")
    parser.add_argument(
        "--items_path",
        required=True,
        help="Путь к каталогу товаров (parquet), например items.parquet",
    )
    parser.add_argument(
        "--matches_path",
        required=True,
        help="Путь к парам кандидатов (parquet), например matches.parquet",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Путь для сохранения предсказаний, например submit.csv",
    )
    return parser.parse_args()


def load_items(items_path):
    return pd.read_parquet(items_path)


def load_matches(matches_path):
    return pd.read_parquet(matches_path)


def load_classifiers():
    """
    Загружает ансамбль ONNX-моделей со всех доступных фолдов.
    Использует onnxruntime (предустановлен в базовом докере).
    """
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    fold_models = sorted(list(WEIGHTS_DIR.glob("catboost_fold_*.onnx")))
    sessions = []

    if fold_models:
        print(f"[инфо] Загрузка ансамбля из {len(fold_models)} ONNX моделей...")
        for model_path in fold_models:
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
            sessions.append(sess)
    else:
        single_model_path = WEIGHTS_DIR / "catboost_model.onnx"
        if single_model_path.exists():
            print(f"[инфо] Загрузка модели {single_model_path}...")
            sess = ort.InferenceSession(str(single_model_path), sess_options=opts, providers=providers)
            sessions.append(sess)
        else:
            raise FileNotFoundError(f"Файлы ONNX весов не найдены в {WEIGHTS_DIR}")

    return sessions


def build_features(items, matches, batch_size=50000):
    """
    Собирает парный DataFrame с суффиксами _1/_2 и батчево вычисляет
    матрицу признаков через build_pair_features() (см. src/feature_builder.py).
    """
    items_indexed = items.set_index("id")
    feature_cols = [c for c in ("name", "title", "attributes", "brand", "category", "price") if c in items.columns]

    total_pairs = len(matches)
    n_batches = (total_pairs + batch_size - 1) // batch_size
    feature_dfs = []

    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, total_pairs)
        b_matches = matches.iloc[b_start:b_end]

        left = items_indexed.loc[b_matches["id1"].values, feature_cols].reset_index(drop=True)
        right = items_indexed.loc[b_matches["id2"].values, feature_cols].reset_index(drop=True)

        pair_df = pd.concat([left.add_suffix("_1"), right.add_suffix("_2")], axis=1)
        batch_feats = build_pair_features(pair_df)
        feature_dfs.append(batch_feats)

    return pd.concat(feature_dfs, ignore_index=True)


def predict(sessions, features):
    """
    Возвращает усредненный непрерывный скор (вероятность класса 'дубликат')
    по ансамблю ONNX моделей для метрики Macro PR-AUC.
    """
    X = features.values.astype(np.float32)
    preds = np.zeros(len(features), dtype=np.float32)

    for sess in sessions:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: X})
        probas = outputs[1]
        if isinstance(probas, list):
            fold_preds = np.fromiter((row[1] for row in probas), dtype=np.float32, count=len(probas))
        else:
            fold_preds = probas[:, 1].astype(np.float32)
        preds += fold_preds / len(sessions)

    return preds


def write_submission(matches, predictions, output_path):
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[инфо] items_path={args.items_path} matches_path={args.matches_path} output_path={args.output_path}")

    items = load_items(args.items_path)
    matches = load_matches(args.matches_path)
    print(f"[инфо] загружено товаров={len(items)}, пар={len(matches)} ({time.perf_counter() - start:.1f}с)")

    classifiers = load_classifiers()
    features = build_features(items, matches)
    print(f"[инфо] признаки готовы, размер: {features.shape} ({time.perf_counter() - start:.1f}с)")

    predictions = predict(classifiers, features)
    write_submission(matches, predictions, args.output_path)
    print(f"[инфо] сабмит записан в {args.output_path}, общее время: {time.perf_counter() - start:.1f}с")


if __name__ == "__main__":
    main()


