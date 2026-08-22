"""
Расчет метрики Macro Averaged PR-AUC по 20 категориям.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def calculate_macro_pr_auc(y_true, y_pred, categories, return_per_category: bool = False):
    """
    Вычисляет соревновательную метрику Macro PR-AUC (Average Precision) по категориям.
    
    Параметры:
        y_true: истинные метки классов (0 или 1), array-like
        y_pred: предсказанные вероятности модели (числа от 0 до 1), array-like
        categories: категории товаров для каждого объекта, array-like
        return_per_category: если True, возвращает также словарь со скорами по каждой категории
        
    Возвращает:
        macro_pr_auc: средний PR-AUC по всем категориям (float)
        (опционально) per_category_scores: словарь {категория: pr_auc}
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "category": np.asarray(categories)
    })
    
    unique_cats = df["category"].unique()
    cat_scores = {}
    
    for cat in unique_cats:
        sub = df[df["category"] == cat]
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values
        
        # Если в категории только один класс (нет позитивных или нет негативных)
        if len(np.unique(y_t)) < 2:
            cat_scores[cat] = 0.0
            continue
            
        score = average_precision_score(y_t, y_p)
        cat_scores[cat] = float(score)
        
    macro_score = float(np.mean(list(cat_scores.values())))
    
    if return_per_category:
        return macro_score, cat_scores
    return macro_score

