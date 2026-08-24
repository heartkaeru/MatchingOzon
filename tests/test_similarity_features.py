import unittest

import pandas as pd

from src.similarity_features import TEXT_SIM_FEATURE_ORDER, build_pair_similarity_features


class SimilarityFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.sample = pd.DataFrame(
            {
                "name_1": [
                    "Samsung Galaxy S21 128GB черный",
                    "Ноутбук игровой Lenovo Legion 16",
                    "Кофе молотый арабика 250 г",
                ],
                "name_2": [
                    "Samsung Galaxy S21 256GB черный",
                    "Lenovo Legion игровой ноутбук 16",
                    "Чай черный листовой 100 г",
                ],
                "attributes_1": [
                    '{"Цвет": "черный", "Экран": "6.2", "Память": "128 ГБ"}',
                    '{"Процессор": "Intel", "ОЗУ": "16 ГБ"}',
                    '{"Вес": "250 г", "Тип": "молотый"}',
                ],
                "attributes_2": [
                    '{"Цвет": "черный", "Экран": "6.2", "Память": "256 ГБ"}',
                    '{"ОЗУ": "16 ГБ", "Процессор": "Intel"}',
                    '{"Вес": "100 г", "Тип": "листовой"}',
                ],
            }
        )
        self.features = build_pair_similarity_features(self.sample)

    def test_feature_order(self):
        self.assertEqual(list(self.features.columns), TEXT_SIM_FEATURE_ORDER)

    def test_name_similarity_features(self):
        self.assertGreater(self.features.loc[0, "name_word_jaccard"], 0.5)
        self.assertGreater(self.features.loc[0, "name_levenshtein_ratio"], 0.8)
        self.assertEqual(self.features.loc[0, "name_first_word_match"], 1)
        self.assertGreater(self.features.loc[1, "name_word_overlap_1_to_2"], 0.9)

    def test_attribute_key_and_value_features(self):
        self.assertEqual(self.features.loc[0, "attrs_common_key_count"], 3)
        self.assertEqual(self.features.loc[0, "attrs_value_match_count"], 2)
        self.assertEqual(self.features.loc[0, "attrs_value_mismatch_count"], 1)
        self.assertEqual(self.features.loc[1, "attrs_value_match_count"], 2)
        self.assertEqual(self.features.loc[1, "attrs_value_mismatch_count"], 0)

    def test_dtypes(self):
        int_cols = [
            "name_first_word_match",
            "attrs_common_key_count",
            "attrs_value_match_count",
            "attrs_value_mismatch_count",
        ]
        self.assertTrue((self.features[int_cols].dtypes == "int16").all())


if __name__ == "__main__":
    unittest.main()
