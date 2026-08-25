import unittest

import pandas as pd

from src.hard_rule_features import HARD_RULE_FEATURE_ORDER, build_pair_hard_features


class HardRuleFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame(
            {
                "name_1": [
                    "Apple iPhone 13 A2633 128GB black",
                    "Кофе молотый 2x500 г, 10 шт",
                    "Блок питания 65W 20V",
                    "Samsung Galaxy S21 SM-G991B 128GB black",
                    "Apple iPhone 13 128GB black",
                    "Xiaomi Redmi Note 13 blue",
                    "Кофе 100 г 1 шт",
                ],
                "name_2": [
                    "Apple iPhone 14 A2882 256 ГБ черный",
                    "Кофе молотый 1000г упаковка 10 штук",
                    "Блок питания 65 Вт 20 В",
                    "Samsung Galaxy S21 SM-G991B 128 ГБ черный",
                    "Samsung iPhone 13 128GB white",
                    "Xiaomi Redmi Note 13 black",
                    "Кофе 1 кг 5 шт",
                ],
                "attributes_1": [
                    '{"color": "black", "RAM": "6 GB"}',
                    "{}",
                    '{"Емкость": "5000 mAh"}',
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                ],
                "attributes_2": [
                    '{"Цвет": "черный", "RAM": "6 ГБ"}',
                    "{}",
                    '{"Емкость": "5000 мАч"}',
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                ],
                "brand_1": [
                    "Apple",
                    None,
                    None,
                    "Samsung",
                    "Apple",
                    None,
                    None,
                ],
                "brand_2": [
                    "Apple",
                    None,
                    None,
                    "Samsung",
                    "Samsung",
                    None,
                    None,
                ],
            }
        )
        self.features = build_pair_hard_features(self.sample)

    def test_feature_order(self):
        self.assertEqual(list(self.features.columns), HARD_RULE_FEATURE_ORDER)

    def test_memory_mismatch(self):
        self.assertEqual(self.features.loc[0, "hard_memory_gb_conflict"], 1)
        self.assertEqual(self.features.loc[0, "is_memory_mismatch"], 1)

    def test_quantity_match_and_mismatch(self):
        self.assertEqual(self.features.loc[1, "hard_weight_g_match"], 1)
        self.assertEqual(self.features.loc[1, "hard_pack_count_match"], 1)
        self.assertEqual(self.features.loc[1, "is_quantity_mismatch"], 0)
        self.assertEqual(self.features.loc[6, "is_quantity_mismatch"], 1)

    def test_power_voltage_and_capacity_match(self):
        self.assertEqual(self.features.loc[2, "hard_power_w_match"], 1)
        self.assertEqual(self.features.loc[2, "hard_voltage_v_match"], 1)
        self.assertEqual(self.features.loc[2, "hard_capacity_mah_match"], 1)

    def test_model_number_flags(self):
        self.assertEqual(self.features.loc[0, "hard_model_conflict"], 1)
        self.assertEqual(self.features.loc[0, "is_model_number_mismatch"], 1)
        self.assertEqual(self.features.loc[3, "hard_model_match"], 1)

    def test_color_and_brand_mismatch(self):
        self.assertEqual(self.features.loc[4, "is_color_mismatch"], 1)
        self.assertEqual(self.features.loc[4, "is_brand_mismatch"], 1)
        self.assertEqual(self.features.loc[5, "is_color_mismatch"], 1)

    def test_mismatch_flags_are_int8(self):
        mismatch_cols = [
            "is_memory_mismatch",
            "is_quantity_mismatch",
            "is_model_number_mismatch",
            "is_color_mismatch",
            "is_brand_mismatch",
        ]
        self.assertTrue((self.features[mismatch_cols].dtypes == "int8").all())


if __name__ == "__main__":
    unittest.main()
