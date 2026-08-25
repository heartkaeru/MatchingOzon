"""
Извлечение признаков для обучающей и валидационной выборок (по 5 фолдам).

Скрипт:
1. Загружает пары со сформированными фолдами из data/processed/matches_folds.parquet
2. Загружает метаданные товаров из data/raw/items_human.parquet
3. Сопоставляет признаки товаров парам (name, attributes, category, etc.)
4. Извлекает полный набор признаков (TF-IDF, строковые сходства, Hard Rules, JSON-атрибуты)
5. Сохраняет готовую матрицу признаков в data/processed/train_features.parquet
"""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Добавление корня репозитория в sys.path для импорта из src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_builder import build_pair_features
from src.text_engine import optimize_pandas_types


def parse_args():
    parser = argparse.ArgumentParser(description="Извлечение признаков для обучения моделей")
    parser.add_argument(
        "--matches_path",
        default="data/processed/matches_folds.parquet",
        help="Путь к парам с фолдами",
    )
    parser.add_argument(
        "--items_path",
        default="data/raw/items_human.parquet",
        help="Путь к каталогу товаров с метаданными",
    )
    parser.add_argument(
        "--output_path",
        default="data/processed/train_features.parquet",
        help="Путь для сохранения матрицы признаков",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Ограничение количества пар (для быстрого тестирования)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Размер батча для пошаговой обработки",
    )
    return parser.parse_args()


def extract_features(
    matches_path: str = "data/processed/matches_folds.parquet",
    items_path: str = "data/raw/items_human.parquet",
    output_path: str = "data/processed/train_features.parquet",
    sample_n: int = None,
    batch_size: int = 50000,
):
    start_time = time.time()
    print("=" * 60)
    print("НАЧАЛО ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 60)

    # 1. Загрузка пар
    print(f"Загрузка пар из {matches_path}...")
    matches = pd.read_parquet(matches_path)
    if sample_n is not None and sample_n < len(matches):
        print(f"Используется сэмпл: {sample_n} пар из {len(matches)}")
        matches = matches.head(sample_n).copy()
    else:
        print(f"Всего пар для обработки: {len(matches)}")

    # 2. Загрузка каталога товаров
    print(f"Загрузка каталога товаров из {items_path}...")
    items = pd.read_parquet(items_path)
    print(f"Загружено {len(items)} товаров. Доступные колонки: {list(items.columns)}")

    # 3. Индексация товаров
    items_indexed = items.set_index("id")
    feature_cols = [
        col for col in ["name", "title", "attributes", "brand", "category", "price"]
        if col in items.columns
    ]

    print(f"Используемые колонки товаров для создания пар: {feature_cols}")

    # 4. Батчевая обработка для экономии памяти и мониторинга прогресса
    total_pairs = len(matches)
    n_batches = (total_pairs + batch_size - 1) // batch_size
    feature_dfs = []

    print(f"\nИзвлечение признаков (батчами по {batch_size} пар, всего батчей: {n_batches})...")
    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, total_pairs)
        b_matches = matches.iloc[b_start:b_end]

        batch_time = time.time()

        # Формирование пар с суффиксами _1 и _2
        left = items_indexed.loc[b_matches["id1"].values, feature_cols].reset_index(drop=True)
        right = items_indexed.loc[b_matches["id2"].values, feature_cols].reset_index(drop=True)
        pair_df = pd.concat([left.add_suffix("_1"), right.add_suffix("_2")], axis=1)

        # Вычисление признаков
        batch_feats = build_pair_features(pair_df)
        feature_dfs.append(batch_feats)

        elapsed = time.time() - batch_time
        print(
            f"  [Батч {b_idx + 1}/{n_batches}] Обработано {b_end}/{total_pairs} пар "
            f"({b_end / total_pairs * 100:.1f}%) за {elapsed:.2f} сек "
            f"({len(b_matches) / max(elapsed, 0.001):.0f} пар/сек)"
        )

    # 5. Объединение признаков
    print("\nСборка итоговой матрицы признаков...")
    all_features = pd.concat(feature_dfs, ignore_index=True)

    # 6. Добавление служебных колонок (id1, id2, target, category, fold)
    meta_cols = ["id1", "id2"]
    for col in ["target", "category", "fold"]:
        if col in matches.columns:
            meta_cols.append(col)

    meta_df = matches[meta_cols].reset_index(drop=True)
    result_df = pd.concat([meta_df, all_features], axis=1)

    # Оптимизация типов данных
    result_df = optimize_pandas_types(result_df)

    # 7. Сохранение результата
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Сохранение в {output_path}...")
    result_df.to_parquet(output_path, index=False)

    total_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Размер датасета: {result_df.shape[0]} строк x {result_df.shape[1]} колонок")
    print(f"Количество сгенерированных фичей: {all_features.shape[1]}")
    print(f"Размер сохраненного файла: {file_size_mb:.2f} МБ")
    print(f"Общее затраченное время: {total_time:.2f} сек")
    print("=" * 60)
    return result_df


def main():
    args = parse_args()
    extract_features(
        matches_path=args.matches_path,
        items_path=args.items_path,
        output_path=args.output_path,
        sample_n=args.sample_n,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()


