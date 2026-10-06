import unittest

import pandas as pd

from src.main import NUMERICAL_COLUMNS, remove_outliers, train_model


def make_measurements():
    rows = []
    for index in range(100):
        age = 18 + (index % 35)
        row = {
            "Gender": 1 + (index % 2),
            "Age": age,
            "TotalHeight": 55 + age * 0.4 + (index % 7) * 0.2,
        }
        for column_index, column in enumerate(NUMERICAL_COLUMNS[1:], start=1):
            row[column] = 10 + (index % 31) + column_index * 0.15
        rows.append(row)

    data = pd.DataFrame(rows)
    for column in NUMERICAL_COLUMNS:
        data.loc[1, column] = 10_000
        data.loc[0, column] = 20_000
    return data


class RegressionPipelineTests(unittest.TestCase):
    def test_outlier_filter_returns_a_copy_without_extreme_training_row(self):
        training = make_measurements().iloc[1:20].copy()
        filtered = remove_outliers(training)

        self.assertNotIn(10_000, filtered[NUMERICAL_COLUMNS].to_numpy())
        self.assertEqual(len(filtered), len(training) - 1)

    def test_missing_measurement_is_kept_for_pipeline_imputation(self):
        training = make_measurements().iloc[2:20].copy()
        missing_index = training.index[0]
        training.loc[missing_index, NUMERICAL_COLUMNS[0]] = pd.NA

        filtered = remove_outliers(training)

        self.assertIn(missing_index, filtered.index)

    def test_test_split_stays_intact_and_outlier_filter_only_reduces_training(self):
        result = train_model(make_measurements())

        self.assertEqual(result["test_rows"], 20)
        self.assertEqual(result["training_rows_before_filter"], 80)
        self.assertLess(
            result["training_rows_after_filter"],
            result["training_rows_before_filter"],
        )
        self.assertEqual(len(result["predictions"]), result["test_rows"])


if __name__ == "__main__":
    unittest.main()
