"""
Быстрые признаки сходства названий и JSON-атрибутов для пары товаров.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


TEXT_SIM_FEATURE_ORDER = [
    "name_word_jaccard",
    "name_char_3gram_jaccard",
    "name_levenshtein_ratio",
    "name_word_overlap_1_to_2",
    "name_word_overlap_2_to_1",
    "name_first_word_match",
    "attrs_key_jaccard",
    "attrs_common_key_count",
    "attrs_value_match_count",
    "attrs_value_mismatch_count",
    "attrs_value_match_ratio",
    "attrs_value_mismatch_ratio",
]

_WORD_RE = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> list[str]:
    return _WORD_RE.findall(_norm_text(value))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = _norm_text(text).replace(" ", "")
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return np.nan
    union = left | right
    if not union:
        return np.nan
    return len(left & right) / len(union)


def _overlap(source: set[Any], target: set[Any]) -> float:
    if not source:
        return np.nan
    return len(source & target) / len(source)


def _flatten_attr_value(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            parts.append(_norm_text(key))
            parts.append(_flatten_attr_value(item))
        return _norm_text(" ".join(parts))
    if isinstance(value, list):
        return _norm_text(" ".join(_flatten_attr_value(item) for item in value))
    return _norm_text(value)


def _parse_attributes(raw: Any) -> dict[str, str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return {}
    else:
        parsed = raw

    if not isinstance(parsed, dict):
        return {}

    attrs = {}
    for key, value in parsed.items():
        key_norm = _norm_text(key)
        if key_norm:
            attrs[key_norm] = _flatten_attr_value(value)
    return attrs


def _name_similarity(left: Any, right: Any) -> dict[str, Any]:
    left_text = _norm_text(left)
    right_text = _norm_text(right)
    left_tokens = set(_tokens(left_text))
    right_tokens = set(_tokens(right_text))
    left_ngrams = _char_ngrams(left_text)
    right_ngrams = _char_ngrams(right_text)

    first_word_match = 0
    left_words = _tokens(left_text)
    right_words = _tokens(right_text)
    if left_words and right_words:
        first_word_match = int(left_words[0] == right_words[0])

    return {
        "name_word_jaccard": _jaccard(left_tokens, right_tokens),
        "name_char_3gram_jaccard": _jaccard(left_ngrams, right_ngrams),
        "name_levenshtein_ratio": fuzz.ratio(left_text, right_text) / 100.0,
        "name_word_overlap_1_to_2": _overlap(left_tokens, right_tokens),
        "name_word_overlap_2_to_1": _overlap(right_tokens, left_tokens),
        "name_first_word_match": first_word_match,
    }


def _attribute_similarity(left_raw: Any, right_raw: Any) -> dict[str, Any]:
    left = _parse_attributes(left_raw)
    right = _parse_attributes(right_raw)
    left_keys = set(left)
    right_keys = set(right)
    common_keys = left_keys & right_keys

    match_count = 0
    mismatch_count = 0
    for key in common_keys:
        if left[key] == right[key]:
            match_count += 1
        else:
            mismatch_count += 1

    common_count = len(common_keys)
    if common_count:
        match_ratio = match_count / common_count
        mismatch_ratio = mismatch_count / common_count
    else:
        match_ratio = np.nan
        mismatch_ratio = np.nan

    return {
        "attrs_key_jaccard": _jaccard(left_keys, right_keys),
        "attrs_common_key_count": common_count,
        "attrs_value_match_count": match_count,
        "attrs_value_mismatch_count": mismatch_count,
        "attrs_value_match_ratio": match_ratio,
        "attrs_value_mismatch_ratio": mismatch_ratio,
    }


def build_pair_similarity_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Строит признаки сходства name/title и attributes для датафрейма с колонками _1/_2.
    """
    name_base = None
    for base in ("name", "title"):
        if f"{base}_1" in pairs.columns and f"{base}_2" in pairs.columns:
            name_base = base
            break
    if name_base is None:
        raise KeyError("Ожидались колонки name_1/name_2 или title_1/title_2")

    attrs_1 = pairs["attributes_1"] if "attributes_1" in pairs.columns else pd.Series(None, index=pairs.index)
    attrs_2 = pairs["attributes_2"] if "attributes_2" in pairs.columns else pd.Series(None, index=pairs.index)

    records = []
    for name_1, name_2, attr_1, attr_2 in zip(
        pairs[f"{name_base}_1"].tolist(),
        pairs[f"{name_base}_2"].tolist(),
        attrs_1.tolist(),
        attrs_2.tolist(),
    ):
        row = {}
        row.update(_name_similarity(name_1, name_2))
        row.update(_attribute_similarity(attr_1, attr_2))
        records.append(row)

    result = pd.DataFrame(records, index=pairs.index, columns=TEXT_SIM_FEATURE_ORDER)
    int_cols = [
        "name_first_word_match",
        "attrs_common_key_count",
        "attrs_value_match_count",
        "attrs_value_mismatch_count",
    ]
    result[int_cols] = result[int_cols].astype("int16")
    float_cols = [col for col in result.columns if col not in int_cols]
    result[float_cols] = result[float_cols].astype("float32")
    return result
