"""
Извлечение признаков для пар товаров (TF-IDF сходство, совпадение брендов/категорий, признаки цен).

Ожидаемая схема датасета пар: колонки с суффиксами _1 / _2 для каждой стороны пары
(name_1/name_2, brand_1/brand_2, category_1/category_2, price_1/price_2).
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.text_engine import clean_text

FEATURE_ORDER = [
    "tfidf_title_sim",
    "brand_match",
    "category_match",
    "price_diff",
    "price_ratio",
]


def compute_tfidf_similarity(df: pd.DataFrame, col1: str, col2: str) -> np.ndarray:
    """
    Построчное косинусное сходство между текстовыми колонками col1 и col2.

    Один TfidfVectorizer обучается на объединении обеих колонок, чтобы словарь
    (и веса IDF) были общими для обеих сторон пары.
    Возвращает массив float32 размерности (len(df),).
    """
    texts1 = df[col1].apply(clean_text).astype(str)
    texts2 = df[col2].apply(clean_text).astype(str)

    if not (texts1.str.strip().any() or texts2.str.strip().any()):
        return np.zeros(len(df), dtype=np.float32)

    combined = pd.concat([texts1, texts2], ignore_index=True)
    vectors = TfidfVectorizer().fit_transform(combined)
    vectors = normalize(vectors)

    n = len(df)
    sims = vectors[:n].multiply(vectors[n:]).sum(axis=1)
    return np.asarray(sims).ravel().astype(np.float32)


def _equality_match(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    1, когда оба поля <prefix>_1 и <prefix>_2 заполнены и равны
    после приведения к нижнему регистру и очистки, иначе 0. Отсутствующие колонки -> все 0.
    """
    col1, col2 = f"{prefix}_1", f"{prefix}_2"
    if col1 not in df.columns or col2 not in df.columns:
        return pd.Series(0, index=df.index, dtype="int8")

    s1 = df[col1].astype("string").str.lower().str.strip()
    s2 = df[col2].astype("string").str.lower().str.strip()
    both_present = s1.notna() & s2.notna()
    equal = ((s1 == s2).fillna(False) & both_present).astype("int8")
    return equal


def _price_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    price_diff = |price_1 - price_2| (сохраняются NaN),
    price_ratio = max/min с NaN для нулевых или пропущенных цен.
    Если колонок нет -> серии из NaN.
    """
    nan_series = lambda: pd.Series(np.nan, index=df.index, dtype="float32")  # noqa: E731

    if "price_1" not in df.columns or "price_2" not in df.columns:
        return nan_series(), nan_series()

    p1 = pd.to_numeric(df["price_1"], errors="coerce")
    p2 = pd.to_numeric(df["price_2"], errors="coerce")

    price_diff = (p1 - p2).abs().astype("float32")

    hi = np.maximum(p1, p2)
    lo = np.minimum(p1, p2)
    price_ratio = pd.Series(
        np.where(lo > 0, hi / lo, np.nan), index=df.index, dtype="float32"
    )
    return price_diff, price_ratio


def build_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Построение матрицы признаков для датасета пар товаров.

    Выходные колонки (фиксированный порядок):
      tfidf_title_sim  - TF-IDF косинусное сходство названий
                         (name_1/name_2, fallback title_1/title_2)
      brand_match      - 1, если бренды указаны и совпадают, иначе 0
      category_match   - аналогичная логика для категорий
      price_diff       - |price_1 - price_2|, NaN сохраняются
      price_ratio      - max(price)/min(price), NaN для нулевых/пропущенных цен
    """
    title_col = None
    for base in ("name", "title"):
        if f"{base}_1" in df.columns and f"{base}_2" in df.columns:
            title_col = base
            break
    if title_col is None:
        raise KeyError("Ожидались колонки name_1/name_2 или title_1/title_2")

    price_diff, price_ratio = _price_features(df)

    features = pd.DataFrame(
        {
            "tfidf_title_sim": compute_tfidf_similarity(df, f"{title_col}_1", f"{title_col}_2"),
            "brand_match": _equality_match(df, "brand"),
            "category_match": _equality_match(df, "category"),
            "price_diff": price_diff,
            "price_ratio": price_ratio,
        },
        columns=FEATURE_ORDER,
    )
    return features


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "name_1": ["Молоко Простоквашино 1л", "Чай зеленый листовой", "Кофе растворимый"],
            "name_2": ["Молоко Простоквашино, 1 л!", "Чай черный байховый", "Кофе растворимый"],
            "brand_1": ["Простоквашино", "Greenfield", None],
            "brand_2": ["простоквашино ", "Akbar", None],
            "category_1": ["Молочные продукты", "Чай", "Кофе"],
            "category_2": ["молочные продукты", "Чай", "Кофе"],
            "price_1": [89.9, 150.0, 0.0],
            "price_2": [89.9, 175.5, 250.0],
        }
    )
    feats = build_pair_features(sample)

    assert list(feats.columns) == FEATURE_ORDER
    assert len(feats) == len(sample)

    # одинаковые заголовки -> сходство близко к 1; частичное перекрытие выше нулевого
    assert abs(feats["tfidf_title_sim"].iloc[2] - 1.0) < 1e-5
    assert feats["tfidf_title_sim"].iloc[0] > feats["tfidf_title_sim"].iloc[1]

    # бренд: совпадение без учета регистра/пробелов; пропущенный бренд -> 0
    assert feats["brand_match"].iloc[0] == 1
    assert feats["brand_match"].iloc[1] == 0
    assert feats["brand_match"].iloc[2] == 0

    # совпадение категорий
    assert feats["category_match"].iloc[0] == 1
    assert feats["category_match"].iloc[1] == 1

    # признаки цены, включая защиту от деления на 0
    assert np.isclose(feats["price_diff"].iloc[1], 25.5)
    assert np.isclose(feats["price_ratio"].iloc[1], 175.5 / 150.0)
    assert np.isnan(feats["price_ratio"].iloc[2])

    # отсутствие опциональных колонок обрабатывается корректно
    degraded = build_pair_features(sample[["name_1", "name_2"]])
    assert list(degraded.columns) == FEATURE_ORDER
    assert (degraded["brand_match"] == 0).all()
    assert degraded["price_diff"].isna().all()

    print("Все тесты успешно пройдены!")

