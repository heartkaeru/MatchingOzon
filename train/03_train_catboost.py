"""
Обучение модели CatBoost на 5 фолдах с Out-Of-Fold (OOF) валидацией по Macro PR-AUC.

Скрипт:
1. Загружает матрицу признаков из data/processed/train_features.parquet
2. Обучает модели CatBoostClassifier на каждом из 5 фолдов
3. Собирает Out-Of-Fold предсказания для всего датасета
4. Рассчитывает официальную соревновательную метрику Macro PR-AUC (общую и по категориям)
5. Анализирует Feature Importance (важность признаков)
6. Сохраняет веса моделей в submission/weights/
"""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Добавление корня репозитория в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import calculate_macro_pr_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение CatBoost с 5-fold Cross-Validation")
    parser.add_argument(
        "--features_path",
        default="data/processed/train_features.parquet",
        help="Путь к матрице признаков",
    )
    parser.add_argument(
        "--weights_dir",
        default="submission/weights",
        help="Директория для сохранения весов моделей",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help="Максимальное количество итераций (деревьев)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Скорость обучения (learning rate)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="Глубина деревьев",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=150,
        help="Количество раундов для ранней остановки",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Количество фолдов",
    )
    parser.add_argument(
        "--task_type",
        default="CPU",
        choices=["CPU", "GPU"],
        help="Устройство для обучения (CPU или GPU)",
    )
    return parser.parse_args()


def train_catboost_cv(
    features_path: str = "data/processed/train_features.parquet",
    weights_dir: str = "submission/weights",
    iterations: int = 2000,
    learning_rate: float = 0.05,
    depth: int = 6,
    early_stopping_rounds: int = 150,
    n_folds: int = 5,
    task_type: str = "CPU",
):
    print("=" * 70)
    print("СТАРТ ОБУЧЕНИЯ CATBOOST (5-FOLD CROSS-VALIDATION)")
    print("=" * 70)

    # 1. Загрузка признаков
    print(f"Загрузка признаков из {features_path}...")
    df = pd.read_parquet(features_path)
    print(f"Загружен датасет: {df.shape[0]} строк x {df.shape[1]} колонок")

    # Исключение мета-колонок из матрицы признаков
    ignore_cols = {"id1", "id2", "target", "category", "fold", "strat_target", "group"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    print(f"Количество признаков для обучения: {len(feature_cols)}")

    os.makedirs(weights_dir, exist_ok=True)

    oof_predictions = np.zeros(len(df), dtype=np.float32)
    feature_importances = np.zeros(len(feature_cols), dtype=np.float64)
    fold_scores = []

    total_start_time = time.time()

    # 2. Обучение по фолдам
    for fold in range(n_folds):
        print(f"\n" + "-" * 50)
        print(f">>> ОБУЧЕНИЕ ФОЛДА {fold + 1}/{n_folds}")
        print("-" * 50)

        train_mask = df["fold"] != fold
        val_mask = df["fold"] == fold

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "target"].astype(int).values

        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, "target"].astype(int).values
        val_cats = df.loc[val_mask, "category"].values

        print(f"Train: {len(X_train)} пар (Pos ratio: {y_train.mean():.3f})")
        print(f"Val:   {len(X_val)} пар (Pos ratio: {y_val.mean():.3f})")

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42 + fold,
            early_stopping_rounds=early_stopping_rounds,
            task_type=task_type,
            verbose=200,
            thread_count=-1,
        )

        fold_start = time.time()
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=200)
        fold_time = time.time() - fold_start

        # Предикты для валидационного фолда
        val_preds = model.predict_proba(val_pool)[:, 1]
        oof_predictions[val_mask] = val_preds

        # Расчет Macro PR-AUC для текущего фолда
        fold_macro_pr_auc = calculate_macro_pr_auc(y_val, val_preds, val_cats)
        fold_scores.append(fold_macro_pr_auc)

        print(
            f"Фолд {fold + 1} завершен за {fold_time:.1f} сек. "
            f"Лучшая итерация: {model.get_best_iteration()}, "
            f"Macro PR-AUC: {fold_macro_pr_auc:.5f}"
        )

        # Сохранение весов фолда
        fold_model_path = os.path.join(weights_dir, f"catboost_fold_{fold}.cbm")
        model.save_model(fold_model_path)
        print(f"Веса фолда сохранены в {fold_model_path}")

        # Если это первый фолд, сохраняем как основную модель для одиночного сабмита
        if fold == 0:
            main_model_path = os.path.join(weights_dir, "catboost_model.cbm")
            model.save_model(main_model_path)

        # Накопление важности признаков
        feature_importances += model.get_feature_importance() / n_folds

    # 3. Итоговая Out-Of-Fold оценка качества
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ОЦЕНКА OUT-OF-FOLD (ПО ВСЕМ 5 ФОЛДАМ)")
    print("=" * 70)

    overall_macro_pr_auc, per_cat_scores = calculate_macro_pr_auc(
        df["target"].values, oof_predictions, df["category"].values, return_per_category=True
    )

    print(f"\n>>> ФИНАЛЬНЫЙ OOF MACRO PR-AUC: {overall_macro_pr_auc:.5f} <<<")
    print(f"Среднее по отдельным фолдам:   {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}\n")

    print("Детализация PR-AUC по категориям:")
    cat_df = pd.DataFrame(
        [{"Категория": cat, "PR-AUC": score} for cat, score in per_cat_scores.items()]
    ).sort_values(by="PR-AUC", ascending=False)
    print(cat_df.to_string(index=False))

    # 4. Топ-20 самых важных признаков
    print("\n" + "-" * 50)
    print("ТОП-20 САМЫХ ВАЖНЫХ ПРИЗНАКОВ (FEATURE IMPORTANCE):")
    print("-" * 50)
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": feature_importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    for rank, row in fi_df.head(20).iterrows():
        print(f"{rank + 1:2d}. {row['Feature']:<35} : {row['Importance']:.3f}%")

    # 5. Сохранение OOF предсказаний
    oof_df = df[["id1", "id2", "category", "fold", "target"]].copy()
    oof_df["oof_pred"] = oof_predictions
    oof_path = "data/processed/oof_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"\nOOF предсказания сохранены в {oof_path}")

    total_time = time.time() - total_start_time
    print(f"\nПолное время валидации и обучения: {total_time:.1f} сек ({total_time / 60:.2f} мин)")
    print("=" * 70)

    return overall_macro_pr_auc, fi_df


def main():
    args = parse_args()
    train_catboost_cv(
        features_path=args.features_path,
        weights_dir=args.weights_dir,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        early_stopping_rounds=args.early_stopping_rounds,
        n_folds=args.n_folds,
        task_type=args.task_type,
    )


if __name__ == "__main__":
    main()


