from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Body Measurements _ original_CSV.csv"
TARGET_COLUMN = "TotalHeight"

NUMERICAL_COLUMNS = [
    "Age",
    "HeadCircumference",
    "ShoulderWidth",
    "ChestWidth",
    "Belly",
    "Waist",
    "Hips",
    "ArmLength",
    "ShoulderToWaist",
    "WaistToKnee",
    "LegLength",
]


def load_data():
    """Load the source dataset and normalize its column names."""
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    required_columns = set(NUMERICAL_COLUMNS) | {"Gender", TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}")

    return data.dropna(subset=[TARGET_COLUMN]).copy()


def remove_outliers(training_data):
    """Filter training rows using IQR bounds calculated from training data only."""
    q1 = training_data[NUMERICAL_COLUMNS].quantile(0.25)
    q3 = training_data[NUMERICAL_COLUMNS].quantile(0.75)
    iqr = q3 - q1
    lower_bounds = q1 - 1.5 * iqr
    upper_bounds = q3 + 1.5 * iqr

    within_bounds = training_data[NUMERICAL_COLUMNS].ge(lower_bounds) & training_data[
        NUMERICAL_COLUMNS
    ].le(upper_bounds)
    missing_values = training_data[NUMERICAL_COLUMNS].isna()
    keep_rows = (within_bounds | missing_values).all(axis=1)
    return training_data.loc[keep_rows].copy()


def build_pipeline():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, ["Gender"]),
        ]
    )


def train_model(data):
    """Split first, filter only training rows, then fit preprocessing and model."""
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    training_data = x_train.copy()
    training_data[TARGET_COLUMN] = y_train
    filtered_training_data = remove_outliers(training_data)

    x_train_filtered = filtered_training_data.drop(columns=[TARGET_COLUMN])
    y_train_filtered = filtered_training_data[TARGET_COLUMN]

    if filtered_training_data.empty:
        raise ValueError("Outlier filtering removed every training row.")

    model = Pipeline(
        [
            ("preprocessor", build_pipeline()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train_filtered, y_train_filtered)

    predictions = model.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    return {
        "model": model,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "training_rows_before_filter": len(x_train),
        "training_rows_after_filter": len(x_train_filtered),
        "test_rows": len(x_test),
    }


def plot_predictions(actual, predicted):
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.7)
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--", color="darkred")
    plt.xlabel("Actual total height (inches)")
    plt.ylabel("Predicted total height (inches)")
    plt.title("Actual vs. Predicted Total Height")
    plt.tight_layout()
    plt.show()


def main():
    data = load_data()
    result = train_model(data)

    print(
        "Training rows after IQR filtering: "
        f"{result['training_rows_after_filter']} / "
        f"{result['training_rows_before_filter']}"
    )
    print(f"Untouched test rows: {result['test_rows']}")
    print(f"Mean Absolute Error (inches): {result['metrics']['mae']:.3f}")
    print(f"Mean Squared Error (inches squared): {result['metrics']['mse']:.3f}")
    print(f"R-squared: {result['metrics']['r2']:.3f}")

    plot_predictions(result["y_test"], result["predictions"])


if __name__ == "__main__":
    main()
