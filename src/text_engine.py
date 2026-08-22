"""
Text cleaning and regex extraction.
"""
import re
import string

import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean input text: lowercase, ё -> е, remove special chars,
    collapse whitespace.

    Non-string input (NaN, None, numbers) -> "".
    Keeps letters (Cyrillic + Latin), digits and single spaces only.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^0-9a-zA-Zа-я ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """
    For every column in text_cols create a cleaned copy named
    f"{col}_clean" (via clean_text). Original columns stay untouched.

    Returns a new DataFrame; input is not modified.
    Missing columns are skipped with no error.
    """
    result = df.copy()
    for col in text_cols:
        if col in result.columns:
            result[f"{col}_clean"] = result[col].apply(clean_text)
    return result


def optimize_pandas_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric dtypes to reduce memory:
      float64 -> float32
      int64   -> int32

    Other dtypes are returned as-is. Returns a new DataFrame.
    """
    result = df.copy()
    float_cols = result.select_dtypes(include=["float64"]).columns
    int_cols = result.select_dtypes(include=["int64"]).columns
    if len(float_cols):
        result[float_cols] = result[float_cols].astype("float32")
    if len(int_cols):
        result[int_cols] = result[int_cols].astype("int32")
    return result


def extract_attributes(text: str) -> dict:
    """
    Regex attribute extractor (volume, weight, quantity, etc.).
    """
    # TODO: Implement attribute extraction (e.g. 500ml, 1kg)
    return {}


if __name__ == "__main__":
    # lowercasing
    assert clean_text("Кофе МОЛОТЫЙ") == "кофе молотый"
    # е/е normalization
    assert clean_text("Ёлка 3Ёх") == "елка 3ех"
    # punctuation / special chars removal
    assert clean_text("Молоко, 3.2% — 1л!") == "молоко 3 2 1л"
    # whitespace collapsing + strip
    assert clean_text("  чай\tзелёный\nлистовой ") == "чай зеленый листовой"
    # non-string inputs
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text("") == ""

    # preprocess_dataframe
    sample = pd.DataFrame({"title": ["Сок ЯБЛОЧНЫЙ", None], "price": [1, 2]})
    processed = preprocess_dataframe(sample, ["title"])
    assert "title_clean" in processed.columns
    assert list(processed["title_clean"]) == ["сок яблочный", ""]
    assert processed["title"].iloc[0] == "Сок ЯБЛОЧНЫЙ"
    assert pd.isna(processed["title"].iloc[1])

    # optimize_pandas_types
    wide = pd.DataFrame(
        {"a": pd.array([1, 2], dtype="int64"), "b": pd.array([1.5, 2.5], dtype="float64")}
    )
    slim = optimize_pandas_types(wide)
    assert str(slim["a"].dtype) == "int32"
    assert str(slim["b"].dtype) == "float32"

    print("all tests passed")
