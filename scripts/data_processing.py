import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Time features
        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])
        X["transaction_hour"] = X["TransactionStartTime"].dt.hour
        X["transaction_day"] = X["TransactionStartTime"].dt.day
        X["transaction_month"] = X["TransactionStartTime"].dt.month
        X["transaction_year"] = X["TransactionStartTime"].dt.year

        # Aggregated features per CustomerId
        agg = X.groupby("CustomerId")["Amount"].agg(
            total_amount="sum",
            avg_amount="mean",
            std_amount="std",
            transaction_count="count"
        ).reset_index()

        X = X.merge(agg, on="CustomerId", how="left")

        return X


def run_pipeline():
    print("Loading data...")
    df = pd.read_csv("data/raw/xente_dataset.csv")

    # Feature Engineering
    print("Running custom feature engineering...")
    fe_pipeline = Pipeline([
        ("features", FeatureEngineer())
    ])
    df_fe = fe_pipeline.fit_transform(df)

    # Separate features
    numeric_features = ["Amount", "Value", "transaction_hour", "transaction_day", "transaction_month",
                        "transaction_year", "total_amount", "avg_amount", "std_amount", "transaction_count"]
    categorical_features = ["ProductCategory", "ChannelId", "PricingStrategy"]

    # Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    print("Transforming dataset...")
    X_processed = preprocessor.fit_transform(df_fe)

    # Convert back to DataFrame
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
                                  columns=feature_names)

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    X_processed_df.to_csv("data/processed/processed_data.csv", index=False)

    print("Feature engineering complete. Processed data saved to data/processed/.")

if name == "main":
    run_pipeline()