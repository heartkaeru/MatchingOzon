"""
Бенчмарк инференса: замер latency, throughput и потребления памяти (GPU/RAM) для пайплайна матчинга.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from submission.run import build_features, load_classifiers, predict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Бенчмарк инференса модели: latency / throughput / память."
    )
    parser.add_argument(
        "--items_path",
        type=str,
        default="data/raw/items_human.parquet",
        help="Путь к каталогу товаров",
    )
    parser.add_argument(
        "--matches_path",
        type=str,
        default="data/processed/matches_folds.parquet",
        help="Путь к парам товаров",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=5000,
        help="Количество пар для замера бенчмарка",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Размер батча для инференса",
    )
    return parser.parse_args()


def report_memory():
    """Пиковое потребление RAM процесса, МБ."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def print_report(results):
    print("\n" + "=" * 50)
    print("           ОТЧЕТ О БЕНЧМАРКЕ ИНФЕРЕНСА")
    print("=" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            formatted = f"{value:.2f}"
        else:
            formatted = str(value)
        print(f" {key:<30}: {formatted}")
    print("=" * 50)


def main():
    start = time.perf_counter()
    args = parse_args()
    print(f"[инфо] Запуск бенчмарка на сэмпле из {args.sample_n} пар...")

    # 1. Загрузка каталога и пар
    t0 = time.perf_counter()
    items = pd.read_parquet(args.items_path)
    matches = pd.read_parquet(args.matches_path).head(args.sample_n)[["id1", "id2"]]
    load_time = time.perf_counter() - t0

    # 2. Загрузка моделей
    t0 = time.perf_counter()
    classifiers = load_classifiers()
    model_load_time = time.perf_counter() - t0

    # 3. Извлечение признаков (Feature Extraction)
    t0 = time.perf_counter()
    features = build_features(items, matches, batch_size=args.batch_size)
    fe_time = time.perf_counter() - t0
    fe_throughput = len(matches) / max(fe_time, 0.001)

    # 4. Предсказание моделей (Inference)
    t0 = time.perf_counter()
    preds = predict(classifiers, features)
    pred_time = time.perf_counter() - t0
    pred_throughput = len(matches) / max(pred_time, 0.001)

    total_time = time.perf_counter() - start
    total_throughput = len(matches) / max(total_time - load_time, 0.001)

    # Оценка времени для public (115k пар) и private (275k пар)
    est_public_min = (115000 / max(total_throughput, 1)) / 60
    est_private_min = (275000 / max(total_throughput, 1)) / 60

    results = {
        "Количество пар": len(matches),
        "Моделей в ансамбле": len(classifiers),
        "Время загрузки данных (сек)": load_time,
        "Время загрузки моделей (сек)": model_load_time,
        "Время генерации фичей (сек)": fe_time,
        "Скорость фичей (пар/сек)": fe_throughput,
        "Время предикта моделей (сек)": pred_time,
        "Скорость предикта (пар/сек)": pred_throughput,
        "Общая скорость пайплайна (пар/сек)": total_throughput,
        "Потребление RAM (МБ)": report_memory(),
        "Прогноз Public 115k (мин, лимит 6м)": est_public_min,
        "Прогноз Private 275k (мин, лимит 13м)": est_private_min,
    }

    print_report(results)


if __name__ == "__main__":
    main()


