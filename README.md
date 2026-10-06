# Body Measurement Height Prediction

An educational regression project that predicts total height from body measurements with scikit-learn's Linear Regression.

## Dataset

The CSV is based on [Body Measurements Datasets](https://data.mendeley.com/datasets/bjv6c9pmp4/1) by Muhammad Kiru (DOI: [10.17632/bjv6c9pmp4.1](https://doi.org/10.17632/bjv6c9pmp4.1)). The measurements are in inches. The source dataset is licensed under CC BY 4.0; its license and attribution are separate from this repository's MIT license.

## Method

- Predict `TotalHeight` from age, body measurements, and gender.
- Split the original rows into an 80% training set and a 20% test set using random seed 42.
- Calculate IQR bounds from the training set only and filter only training rows. The test rows remain untouched.
- Fit missing-value imputation, feature scaling, and gender encoding within the training pipeline.
- Evaluate once on the held-out test set using MAE, MSE, and R-squared.

The current reference run reports MAE 5.475 inches, MSE 57.004 inches squared, and R-squared 0.637. These values describe this one fixed split and are not a guarantee of generalization.

## Requirements and run

Python 3.9 or newer:

```bash
python -m pip install -r requirements.txt
python src/main.py
```

The script prints the training and test sample counts and evaluation metrics, then displays an actual-versus-predicted plot.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
linear-regression-project/
├── data/
│   └── Body Measurements _ original_CSV.csv
├── src/
│   └── main.py
├── tests/
│   └── test_pipeline.py
├── LICENSE
├── README.md
└── requirements.txt
```
