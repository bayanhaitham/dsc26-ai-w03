"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
    "customer_prior_parcels",
    "customer_prior_fail_rate",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer history using only strictly earlier parcels."""
    result = df.copy()

    # Keep track of the original row positions.
    work = df[["customer_id", "created_at", "failed"]].copy()
    work["_row_position"] = range(len(work))

    # Group parcels by customer and timestamp.
    # Parcels with identical timestamps are treated as one group.
    history = (
        work.groupby(
            ["customer_id", "created_at"],
            dropna=False,
            sort=True
        )["failed"]
        .agg(
            parcel_count="size",
            failure_count="sum"
        )
        .reset_index()
    )

    # Calculate cumulative history for each customer.
    customer_groups = history.groupby(
        "customer_id",
        dropna=False,
        sort=False
    )

    history["customer_prior_parcels"] = (
        customer_groups["parcel_count"].cumsum()
        - history["parcel_count"]
    )

    prior_failures = (
        customer_groups["failure_count"].cumsum()
        - history["failure_count"]
    )

    # Avoid division by zero for customers with no previous parcels.
    history["customer_prior_fail_rate"] = (
        prior_failures
        / history["customer_prior_parcels"].replace(0, float("nan"))
    )

    # Attach the history features to the original rows.
    merged = work.merge(
        history[
            [
                "customer_id",
                "created_at",
                "customer_prior_parcels",
                "customer_prior_fail_rate"
            ]
        ],
        on=["customer_id", "created_at"],
        how="left",
        sort=False
    )

    # Restore the original row order.
    merged = merged.sort_values("_row_position")

    result["customer_prior_parcels"] = (
        merged["customer_prior_parcels"].to_numpy()
    )
    result["customer_prior_fail_rate"] = (
        merged["customer_prior_fail_rate"].to_numpy()
    )

    return result


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    n_test = math.ceil(len(df) * test_fraction)

    sorted_df = df.sort_values(time_col)

    boundary = sorted_df.iloc[-n_test:][time_col].min()

    train = sorted_df[sorted_df[time_col] < boundary].copy()
    test = sorted_df[sorted_df[time_col] >= boundary].copy()

    return train, test

def clean_text(X):
    """Convert address text into a clean 1D array."""
    return pd.Series(X).fillna("").astype(str).to_numpy()


def extract_datetime_features(X):
    """Extract useful numerical features from created_at."""
    dates = pd.to_datetime(X.iloc[:, 0], errors="coerce")

    return np.column_stack([
        dates.dt.hour,
        dates.dt.dayofweek,
        dates.dt.day,
    ])
def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an unfitted pipeline for predicting parcel failure."""

    numeric_features = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]

    categorical_features = [
        "customer_id",
        "merchant_id",
        "area",
        "promised_slot",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    text_pipeline = Pipeline([
        ("clean", FunctionTransformer(clean_text, validate=False)),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2
        )),
    ])

    datetime_pipeline = Pipeline([
        ("extract", FunctionTransformer(
            extract_datetime_features,
            validate=False
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("text", text_pipeline, "address_text"),
            ("datetime", datetime_pipeline, ["created_at"]),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=seed
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])



def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate a trained model using the test dataset."""

    # Get the probability of failure (class 1).
    predictions = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    # Get the actual outcomes.
    actual = test_df[TARGET]

    return {
        "roc_auc": roc_auc_score(actual, predictions),
        "pr_auc": average_precision_score(actual, predictions),
        "brier": brier_score_loss(actual, predictions),
        "base_rate": actual.mean(),
        "n": len(test_df),
    }




def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0
) -> dict:
    """Compare our model against a simple dummy baseline."""

    # Build and train our machine learning model.
    model = build_pipeline(seed)
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])

    # Build and train a simple baseline model.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])

    # Evaluate both models on the test data.
    model_results = evaluate(model, test_df)
    dummy_results = evaluate(dummy, test_df)

    # Calculate the improvement in PR-AUC.
    lift_pr_auc = model_results["pr_auc"] - dummy_results["pr_auc"]

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": lift_pr_auc,
    }

