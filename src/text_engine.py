"""
Очистка текста и извлечение характеристик с помощью регулярных выражений (regex).
"""
import re
import string

import pandas as pd


def clean_text(text: str) -> str:
    """
    Очистка входного текста: приведение к нижнему регистру, ё -> е,
    удаление специальных символов, схлопывание пробелов.

    Нестроковый ввод (NaN, None, числа) -> "".
    Сохраняет только буквы (кириллица + латиница), цифры и одиночные пробелы.
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
    Для каждой колонки из text_cols создает очищенную копию с именем
    f"{col}_clean" (через clean_text). Исходные колонки остаются без изменений.

    Возвращает новый DataFrame; входной датафрейм не мутирует.
    Отсутствующие колонки пропускаются без ошибок.
    """
    result = df.copy()
    for col in text_cols:
        if col in result.columns:
            result[f"{col}_clean"] = result[col].apply(clean_text)
    return result


def optimize_pandas_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Уменьшение разрядности числовых типов для экономии памяти:
      float64 -> float32
      int64   -> int32

    Остальные типы данных остаются без изменений. Возвращает новый DataFrame.
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
    Извлечение числовых характеристик из текста (объем, вес, количество штук в упаковке, память и т.д.).
    """
    # TODO: Реализовать regex-парсинг (например, 500мл, 1кг, 128gb, 2 шт)
    return {}


if __name__ == "__main__":
    # приведение к нижнему регистру
    assert clean_text("Кофе МОЛОТЫЙ") == "кофе молотый"
    # нормализация ё/е
    assert clean_text("Ёлка 3Ёх") == "елка 3ех"
    # удаление пунктуации и спецсимволов
    assert clean_text("Молоко, 3.2% — 1л!") == "молоко 3 2 1л"
    # схлопывание пробелов и strip
    assert clean_text("  чай\tзелёный\nлистовой ") == "чай зеленый листовой"
    # нестроковый ввод
    assert clean_text(None) == ""
    assert clean_text(123) == ""
    assert clean_text("") == ""

    # предобработка датафрейма (preprocess_dataframe)
    sample = pd.DataFrame({"title": ["Сок ЯБЛОЧНЫЙ", None], "price": [1, 2]})
    processed = preprocess_dataframe(sample, ["title"])
    assert "title_clean" in processed.columns
    assert list(processed["title_clean"]) == ["сок яблочный", ""]
    assert processed["title"].iloc[0] == "Сок ЯБЛОЧНЫЙ"
    assert pd.isna(processed["title"].iloc[1])

    # оптимизация типов данных (optimize_pandas_types)
    wide = pd.DataFrame(
        {"a": pd.array([1, 2], dtype="int64"), "b": pd.array([1.5, 2.5], dtype="float64")}
    )
    slim = optimize_pandas_types(wide)
    assert str(slim["a"].dtype) == "int32"
    assert str(slim["b"].dtype) == "float32"

    print("Все тесты успешно пройдены!")

