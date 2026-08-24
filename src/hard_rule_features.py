"""
Быстрые regex-признаки и hard rules для матчинга товаров.

Модуль намеренно легкий: использует только регулярные выражения, json-парсинг
и pandas, поэтому подходит для жестких лимитов времени на инференсе.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_SPECS = {
    "weight_g": 0.03,
    "volume_ml": 0.03,
    "memory_gb": 0.01,
    "pack_count": 0.0,
    "power_w": 0.03,
    "voltage_v": 0.03,
    "capacity_mah": 0.03,
    "dimension_mm": 0.03,
}

HARD_RULE_FEATURE_ORDER = [
    "hard_model_match",
    "hard_model_conflict",
    "hard_model_jaccard",
    "hard_color_match",
    "hard_color_conflict",
    "hard_number_jaccard",
    "hard_number_conflict",
]

for _name in NUMERIC_SPECS:
    HARD_RULE_FEATURE_ORDER.extend(
        [
            f"hard_{_name}_both",
            f"hard_{_name}_match",
            f"hard_{_name}_conflict",
            f"hard_{_name}_abs_diff",
            f"hard_{_name}_rel_diff",
            f"hard_{_name}_ratio",
        ]
    )


_DECIMAL = r"\d+(?:[,.]\d+)?"
_MULTIPLY = r"[xх*]"

_UNIT_ALIASES = {
    "weight_g": {
        "mg": 0.001,
        "мг": 0.001,
        "kg": 1000.0,
        "кг": 1000.0,
        "g": 1.0,
        "гр": 1.0,
        "г": 1.0,
    },
    "volume_ml": {
        "ml": 1.0,
        "мл": 1.0,
        "l": 1000.0,
        "л": 1000.0,
    },
    "memory_gb": {
        "gb": 1.0,
        "гб": 1.0,
        "gib": 1.0,
        "гбайт": 1.0,
        "tb": 1024.0,
        "тб": 1024.0,
        "tб": 1024.0,
        "тбайт": 1024.0,
        "mb": 1.0 / 1024.0,
        "мб": 1.0 / 1024.0,
    },
    "power_w": {
        "kw": 1000.0,
        "квт": 1000.0,
        "w": 1.0,
        "вт": 1.0,
    },
    "voltage_v": {
        "v": 1.0,
        "в": 1.0,
    },
    "capacity_mah": {
        "mah": 1.0,
        "мач": 1.0,
        "маh": 1.0,
        "мaч": 1.0,
        "ma h": 1.0,
        "ма ч": 1.0,
    },
    "dimension_mm": {
        "mm": 1.0,
        "мм": 1.0,
        "cm": 10.0,
        "см": 10.0,
        "m": 1000.0,
        "м": 1000.0,
    },
}

_ALL_UNITS = sorted(
    {unit for aliases in _UNIT_ALIASES.values() for unit in aliases},
    key=len,
    reverse=True,
)
_UNIT_PATTERN = "|".join(re.escape(unit) for unit in _ALL_UNITS)
_NUMBER_UNIT_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_DECIMAL})\s*(?P<unit>{_UNIT_PATTERN})(?![\w])",
    flags=re.IGNORECASE,
)
_MULTIPACK_UNIT_RE = re.compile(
    rf"(?<![\w.])(?P<count>\d{{1,4}})\s*{_MULTIPLY}\s*"
    rf"(?P<num>{_DECIMAL})\s*(?P<unit>{_UNIT_PATTERN})(?![\w])",
    flags=re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    rf"(?<![\w.])(?P<a>{_DECIMAL})\s*{_MULTIPLY}\s*"
    rf"(?P<b>{_DECIMAL})(?:\s*{_MULTIPLY}\s*(?P<c>{_DECIMAL}))?\s*"
    r"(?P<unit>мм|mm|см|cm|м|m)(?![\w])",
    flags=re.IGNORECASE,
)
_COUNT_RE = re.compile(
    rf"(?<![\w.])(?P<count>\d{{1,4}})\s*(?:шт|штук|pcs|pc|pieces|pack|packs|уп|упак|упаковк[аи]?)(?![\w])",
    flags=re.IGNORECASE,
)
_PACK_OF_RE = re.compile(r"pack\s+of\s+(?P<count>\d{1,4})(?![\w])", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[,.]\d+)?(?![\w])")
_MODEL_RE = re.compile(
    r"(?<![\w])(?=[A-ZА-Я0-9-]{3,24}(?![\w]))"
    r"(?=[A-ZА-Я0-9-]*\d)(?=[A-ZА-Я0-9-]*[A-ZА-Я])"
    r"[A-ZА-Я]{1,8}-?\d[A-ZА-Я0-9-]{0,16}(?![\w])"
)
_MODEL_HYPHEN_RE = re.compile(
    r"(?<![\w])(?=[A-ZА-Я0-9-]{4,30}(?![\w]))"
    r"(?=[A-ZА-Я0-9-]*\d)(?=[A-ZА-Я0-9-]*[A-ZА-Я])"
    r"[A-ZА-Я0-9]{1,10}(?:-[A-ZА-Я0-9]{1,18})+(?![\w])"
)
_MODEL_WORD_RE = re.compile(
    r"(?i)\b(?:iphone|ipad|galaxy|redmi|poco|realme|honor|mate|nova|pixel|watch|airpods|"
    r"macbook|thinkpad|ideapad|aspire|vivobook|playstation|xbox)\s*"
    r"(?P<model>[a-zа-я]?\d{1,3}(?:\s*(?:pro|max|plus|mini|ultra|se|fe|s|x|c))?)\b"
)

_COLOR_ALIASES = {
    "black": ("black", "черный", "черная", "черное", "черн", "чёрный", "чёрная", "чёрное"),
    "white": ("white", "белый", "белая", "белое", "бел"),
    "blue": ("blue", "синий", "синяя", "синее", "голубой", "голубая", "navy"),
    "red": ("red", "красный", "красная", "красное", "бордовый", "бордо"),
    "green": ("green", "зеленый", "зеленая", "зеленое", "зелёный", "зелёная", "зелёное"),
    "yellow": ("yellow", "желтый", "желтая", "желтое", "жёлтый", "жёлтая", "жёлтое"),
    "gray": ("gray", "grey", "серый", "серая", "серое", "графит", "graphite"),
    "silver": ("silver", "серебристый", "серебро", "silver"),
    "gold": ("gold", "золотой", "золотистый", "золото"),
    "pink": ("pink", "розовый", "розовая", "розовое"),
    "purple": ("purple", "violet", "фиолетовый", "фиолетовая", "сиреневый"),
    "brown": ("brown", "коричневый", "коричневая", "шоколадный"),
    "orange": ("orange", "оранжевый", "оранжевая"),
    "beige": ("beige", "бежевый", "беж", "кремовый"),
    "transparent": ("transparent", "прозрачный", "прозрачная"),
}
_COLOR_LOOKUP = {
    alias: color for color, aliases in _COLOR_ALIASES.items() for alias in aliases
}
_COLOR_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(a) for a in sorted(_COLOR_LOOKUP, key=len, reverse=True)) + r")(?![\w])",
    flags=re.IGNORECASE,
)

_MODEL_STOP_UNITS = {u.lower().replace(" ", "") for u in _ALL_UNITS}


def _to_float(value: str) -> float:
    return float(value.replace(",", "."))


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _flatten_json(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_json(item)
    elif value is not None:
        yield str(value)


def _parse_attributes(raw: Any) -> tuple[str, dict[str, str]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "", {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return "", {}
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return stripped, {}
    else:
        parsed = raw

    if isinstance(parsed, dict):
        flat = " ".join(_flatten_json(parsed))
        direct = {str(k).lower(): _norm_text(v) for k, v in parsed.items()}
        return flat, direct
    if isinstance(parsed, list):
        return " ".join(_flatten_json(parsed)), {}
    return _norm_text(parsed), {}


def _unit_kind(unit: str) -> tuple[str | None, float]:
    unit_norm = unit.lower().replace(" ", "")
    for kind, aliases in _UNIT_ALIASES.items():
        normalized = {k.replace(" ", ""): v for k, v in aliases.items()}
        if unit_norm in normalized:
            return kind, normalized[unit_norm]
    return None, 1.0


def _best(values: list[float]) -> float:
    if not values:
        return np.nan
    if len(values) == 1:
        return float(values[0])
    return float(max(values))


def _extract_numbers(text: str) -> set[str]:
    result = set()
    occupied: list[tuple[int, int]] = []
    for match in _NUMBER_UNIT_RE.finditer(text):
        occupied.append(match.span("num"))
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        if any(start >= lo and end <= hi for lo, hi in occupied):
            continue
        result.add(match.group(0).replace(",", ".").lstrip("0") or "0")
    return result


def _extract_models(text: str) -> set[str]:
    upper_text = text.upper()
    models = set()
    for match in _MODEL_HYPHEN_RE.finditer(upper_text):
        models.add(match.group(0).strip("-"))
    for match in _MODEL_RE.finditer(upper_text):
        token = match.group(0).strip("-")
        compact = token.replace("-", "").lower()
        if compact in _MODEL_STOP_UNITS:
            continue
        models.add(token)
    for match in _MODEL_WORD_RE.finditer(text):
        models.add(match.group(0).lower().replace(" ", ""))
    return models


def _extract_colors(text: str, attrs: dict[str, str]) -> set[str]:
    candidates = [text]
    for key, value in attrs.items():
        if any(marker in key for marker in ("цвет", "color", "colour")):
            candidates.insert(0, value)

    colors = set()
    for candidate in candidates:
        for match in _COLOR_RE.finditer(candidate):
            colors.add(_COLOR_LOOKUP[match.group(1).lower()])
    return colors


def extract_item_hard_features(name: Any, attributes: Any = None) -> dict[str, Any]:
    """
    Извлекает нормализованные жесткие атрибуты из name и JSON attributes.
    """
    attr_text, attr_dict = _parse_attributes(attributes)
    text = f"{_norm_text(name)} {attr_text}".lower().replace("ё", "е")
    values: dict[str, list[float]] = {name: [] for name in NUMERIC_SPECS}

    for match in _DIMENSION_RE.finditer(text):
        kind, factor = _unit_kind(match.group("unit"))
        if kind == "dimension_mm":
            dims = [_to_float(match.group("a")), _to_float(match.group("b"))]
            if match.group("c"):
                dims.append(_to_float(match.group("c")))
            values["dimension_mm"].append(max(dims) * factor)

    for match in _MULTIPACK_UNIT_RE.finditer(text):
        count = float(match.group("count"))
        kind, factor = _unit_kind(match.group("unit"))
        if kind in values:
            values[kind].append(count * _to_float(match.group("num")) * factor)
        values["pack_count"].append(count)

    for match in _NUMBER_UNIT_RE.finditer(text):
        kind, factor = _unit_kind(match.group("unit"))
        if kind in values:
            values[kind].append(_to_float(match.group("num")) * factor)

    for match in _COUNT_RE.finditer(text):
        values["pack_count"].append(float(match.group("count")))
    for match in _PACK_OF_RE.finditer(text):
        values["pack_count"].append(float(match.group("count")))

    features = {key: _best(vals) for key, vals in values.items()}
    features["colors"] = tuple(sorted(_extract_colors(text, attr_dict)))
    features["models"] = tuple(sorted(_extract_models(f"{_norm_text(name)} {attr_text}")))
    features["numbers"] = tuple(sorted(_extract_numbers(text)))
    return features


def build_item_hard_features(items: pd.DataFrame) -> pd.DataFrame:
    """
    Строит item-level жесткие атрибуты для датафрейма каталога.
    Обязательная колонка: name. Опциональная колонка: attributes.
    """
    if "name" not in items.columns:
        raise KeyError("Ожидалась колонка товара 'name'")

    attrs = items["attributes"] if "attributes" in items.columns else pd.Series(None, index=items.index)
    records = [
        extract_item_hard_features(name, attr)
        for name, attr in zip(items["name"].tolist(), attrs.tolist())
    ]
    return pd.DataFrame(records, index=items.index)


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return np.nan
    union = left | right
    if not union:
        return np.nan
    return len(left & right) / len(union)


def _set_pair_features(left: Any, right: Any) -> tuple[int, int, float]:
    left_set = set(left) if isinstance(left, (tuple, list, set)) else set()
    right_set = set(right) if isinstance(right, (tuple, list, set)) else set()
    if not left_set or not right_set:
        return 0, 0, np.nan
    match = int(bool(left_set & right_set))
    conflict = int(not match)
    return match, conflict, float(_jaccard(left_set, right_set))


def _numeric_pair_features(left: Any, right: Any, tolerance: float) -> tuple[int, int, int, float, float, float]:
    left_value = pd.to_numeric(left, errors="coerce")
    right_value = pd.to_numeric(right, errors="coerce")
    if pd.isna(left_value) or pd.isna(right_value):
        return 0, 0, 0, np.nan, np.nan, np.nan

    left_value = float(left_value)
    right_value = float(right_value)
    abs_diff = abs(left_value - right_value)
    denom = max(abs(left_value), abs(right_value), 1e-9)
    rel_diff = abs_diff / denom
    ratio = max(abs(left_value), abs(right_value)) / max(min(abs(left_value), abs(right_value)), 1e-9)
    match = int(rel_diff <= tolerance)
    return 1, match, int(not match), abs_diff, rel_diff, ratio


def build_pair_hard_features_from_item_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Строит pair-level hard-rule признаки из двух выровненных item-feature датафреймов.
    """
    out: dict[str, list[Any]] = {col: [] for col in HARD_RULE_FEATURE_ORDER}

    for left_row, right_row in zip(left.to_dict("records"), right.to_dict("records")):
        model_match, model_conflict, model_jaccard = _set_pair_features(
            left_row.get("models"), right_row.get("models")
        )
        color_match, color_conflict, _ = _set_pair_features(
            left_row.get("colors"), right_row.get("colors")
        )
        _, number_conflict, number_jaccard = _set_pair_features(
            left_row.get("numbers"), right_row.get("numbers")
        )

        out["hard_model_match"].append(model_match)
        out["hard_model_conflict"].append(model_conflict)
        out["hard_model_jaccard"].append(model_jaccard)
        out["hard_color_match"].append(color_match)
        out["hard_color_conflict"].append(color_conflict)
        out["hard_number_jaccard"].append(number_jaccard)
        out["hard_number_conflict"].append(number_conflict)

        for name, tolerance in NUMERIC_SPECS.items():
            both, match, conflict, abs_diff, rel_diff, ratio = _numeric_pair_features(
                left_row.get(name), right_row.get(name), tolerance
            )
            out[f"hard_{name}_both"].append(both)
            out[f"hard_{name}_match"].append(match)
            out[f"hard_{name}_conflict"].append(conflict)
            out[f"hard_{name}_abs_diff"].append(abs_diff)
            out[f"hard_{name}_rel_diff"].append(rel_diff)
            out[f"hard_{name}_ratio"].append(ratio)

    result = pd.DataFrame(out, index=left.index)
    int_cols = [c for c in result.columns if c.endswith(("_both", "_match", "_conflict"))]
    result[int_cols] = result[int_cols].astype("int8")
    float_cols = [c for c in result.columns if c not in int_cols]
    result[float_cols] = result[float_cols].astype("float32")
    return result[HARD_RULE_FEATURE_ORDER]


def build_pair_hard_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Строит pair-level hard-rule признаки из датафрейма с колонками _1/_2.
    Обязательные колонки: name_1/name_2 или title_1/title_2.
    Опциональные колонки: attributes_1/attributes_2.
    """
    name_base = None
    for base in ("name", "title"):
        if f"{base}_1" in pairs.columns and f"{base}_2" in pairs.columns:
            name_base = base
            break
    if name_base is None:
        raise KeyError("Ожидались колонки name_1/name_2 или title_1/title_2")

    left = pd.DataFrame(
        {
            "name": pairs[f"{name_base}_1"],
            "attributes": pairs["attributes_1"] if "attributes_1" in pairs.columns else None,
        },
        index=pairs.index,
    )
    right = pd.DataFrame(
        {
            "name": pairs[f"{name_base}_2"],
            "attributes": pairs["attributes_2"] if "attributes_2" in pairs.columns else None,
        },
        index=pairs.index,
    )
    return build_pair_hard_features_from_item_features(
        build_item_hard_features(left),
        build_item_hard_features(right),
    )


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "name_1": [
                "Apple iPhone 13 A2633 128GB black",
                "Кофе молотый 2x500 г, 10 шт",
                "Блок питания 65W 20V",
                "Samsung Galaxy S21 SM-G991B 128GB black",
            ],
            "name_2": [
                "Apple iPhone 14 A2882 256 ГБ черный",
                "Кофе молотый 1000г упаковка 10 штук",
                "Блок питания 65 Вт 20 В",
                "Samsung Galaxy S21 SM-G991B 128 ГБ черный",
            ],
            "attributes_1": [
                '{"color": "black", "RAM": "6 GB"}',
                "{}",
                '{"Емкость": "5000 mAh"}',
                "{}",
            ],
            "attributes_2": [
                '{"Цвет": "черный", "RAM": "6 ГБ"}',
                "{}",
                '{"Емкость": "5000 мАч"}',
                "{}",
            ],
        }
    )
    feats = build_pair_hard_features(sample)
    assert list(feats.columns) == HARD_RULE_FEATURE_ORDER
    assert feats.loc[0, "hard_color_match"] == 1
    assert feats.loc[0, "hard_memory_gb_conflict"] == 1
    assert feats.loc[0, "hard_model_conflict"] == 1
    assert feats.loc[1, "hard_weight_g_match"] == 1
    assert feats.loc[1, "hard_pack_count_match"] == 1
    assert feats.loc[2, "hard_power_w_match"] == 1
    assert feats.loc[2, "hard_voltage_v_match"] == 1
    assert feats.loc[2, "hard_capacity_mah_match"] == 1
    assert feats.loc[3, "hard_model_match"] == 1
    print("Проверки hard-rule признаков пройдены")
