from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Body Measurements _ original_CSV.csv"
OUTPUT_PATH = BASE_DIR / "data" / "preprocessed_data.csv"


NUMERICAL_COLUMNS = [
    "Age",
    "HeadCircumference",
    "ShoulderWidth",
    "ChestWidth ",
    "Belly ",
    "Waist ",
    "Hips ",
    "ArmLength ",
    "ShoulderToWaist ",
    "WaistToKnee ",
    "LegLength",
]


def load_data():
    return pd.read_csv(DATA_PATH)


def remove_outliers(data):
    q1 = data[NUMERICAL_COLUMNS].quantile(0.25)
    q3 = data[NUMERICAL_COLUMNS].quantile(0.75)
    iqr = q3 - q1

    mask = ~(
        (data[NUMERICAL_COLUMNS] < (q1 - 1.5 * iqr))
        | (data[NUMERICAL_COLUMNS] > (q3 + 1.5 * iqr))
    ).any(axis=1)

    return data[mask]


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
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERICAL_COLUMNS),
            ("categorical", categorical_pipeline, ["Gender"]),
        ]
    )

    return preprocessor


def train_model(data):
    x = data.drop("TotalHeight", axis=1)
    y = data["TotalHeight"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            ("preprocessor", build_pipeline()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Mean Squared Error:", mean_squared_error(y_test, predictions))
    print("R-squared:", r2_score(y_test, predictions))

    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Total Height")
    plt.ylabel("Predicted Total Height")
    plt.title("Actual vs Predicted Total Height")
    plt.show()


def main():
    data = load_data()

    print(data.head())
    print(data.info())

    data = remove_outliers(data)
    data.to_csv(OUTPUT_PATH, index=False)

    train_model(data)


if __name__ == "__main__":
    main()