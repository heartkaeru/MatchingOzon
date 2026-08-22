"""
Графовое разбиение на фолды без утечек данных (StratifiedGroupKFold по компонентам связности).
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold


def split_data(
    matches_path: str = "data/raw/matches.parquet",
    items_path: str = "data/raw/items_human.parquet",
    output_path: str = "data/processed/matches_folds.parquet",
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Выполняет разбиение пар на фолды кросс-валидации на основе графа компонент связности.
    Исключает утечки данных между обучающей и валидационной выборками.
    """
    print(f"Загрузка пар товаров из {matches_path}...")
    matches = pd.read_parquet(matches_path)
    print(f"Всего пар в разметке: {len(matches)}")

    print(f"Загрузка метаданных товаров из {items_path}...")
    items = pd.read_parquet(items_path, columns=["id", "category"])
    
    # Сопоставление категорий парам товаров
    print("Сопоставление категорий парам...")
    cat_map = items.set_index("id")["category"].to_dict()
    matches["category"] = matches["id1"].map(cat_map)

    # Построение графа связей между товарами
    print("Построение графа компонент связности...")
    all_ids = np.unique(matches[["id1", "id2"]].values)
    id_to_idx = {item_id: idx for idx, item_id in enumerate(all_ids)}

    row = matches["id1"].map(id_to_idx).values
    col = matches["id2"].map(id_to_idx).values
    data = np.ones(len(matches), dtype=np.int32)
    n_nodes = len(all_ids)

    adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    n_components, item_components = connected_components(adj_matrix, directed=False)
    print(f"Вершин в графе (уникальных товаров): {n_nodes}, Компонент связности: {n_components}")

    # Сопоставление номера компоненты каждой паре (группа для GroupKFold)
    item_to_comp = dict(zip(all_ids, item_components))
    matches["group"] = matches["id1"].map(item_to_comp)

    # Создание метки для стратификации (категория + таргет)
    matches["strat_target"] = matches["category"].astype(str) + "_" + matches["target"].astype(str)

    # Разбиение на фолды с сохранением баланса классов и целостности компонент
    print(f"Разбиение на {n_splits} фолдов с помощью StratifiedGroupKFold...")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    matches["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(matches, y=matches["strat_target"], groups=matches["group"])
    ):
        matches.loc[val_idx, "fold"] = fold

    # Строгая проверка на отсутствие утечек данных между Train и Val
    print("\n--- Проверка отсутствия утечек данных (Data Leakage) ---")
    for fold in range(n_splits):
        train_pairs = matches[matches["fold"] != fold]
        val_pairs = matches[matches["fold"] == fold]
        
        train_ids = set(train_pairs["id1"]).union(set(train_pairs["id2"]))
        val_ids = set(val_pairs["id1"]).union(set(val_pairs["id2"]))
        
        overlap = train_ids.intersection(val_ids)
        assert len(overlap) == 0, f"ОБНАРУЖЕНА УТЕЧКА на фолде {fold}! Количество пересечений: {len(overlap)}"
        
        pos_ratio = val_pairs["target"].mean()
        n_cats = val_pairs["category"].nunique()
        print(
            f"Фолд {fold}: {len(val_pairs)} пар ({len(val_pairs)/len(matches)*100:.1f}%), "
            f"доля позитивного класса: {pos_ratio*100:.2f}%, уникальных категорий: {n_cats}, пересечений ID: {len(overlap)}"
        )

    # Сохранение обработанных фолдов
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_cols = ["id1", "id2", "target", "category", "fold"]
    matches[save_cols].to_parquet(output_path, index=False)
    print(f"\nФолды успешно сохранены в {output_path}!")


if __name__ == "__main__":
    split_data()


