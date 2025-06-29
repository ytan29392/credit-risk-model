import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def calculate_rfm(df):
    # Convert date
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Set snapshot date (max date + 1)
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    # RFM Calculation
    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).rename(columns={
        "TransactionStartTime": "Recency",
        "TransactionId": "Frequency",
        "Amount": "Monetary"
    }).reset_index()

    return rfm

def cluster_rfm(rfm_df, n_clusters=3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Label high-risk cluster: lowest frequency & monetary, highest recency
    cluster_stats = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_stats["Frequency"].idxmin()  # Lowest freq = least active

    rfm_df["is_high_risk"] = (rfm_df["Cluster"] == high_risk_cluster).astype(int)
    return rfm_df[["CustomerId", "is_high_risk"]]

def merge_target(df, target_df):
    return df.merge(target_df, on="CustomerId", how="left")

def main():
    print("Loading data...")
    df = pd.read_csv("data/raw/xente_dataset.csv")

    print("Calculating RFM...")
    rfm_df = calculate_rfm(df)

    print("Clustering customers...")
    target_df = cluster_rfm(rfm_df)

    print("🪄 Merging target variable...")
    labeled = merge_target(df, target_df)

    os.makedirs("data/processed", exist_ok=True)
    labeled.to_csv("data/processed/labeled_data.csv", index=False)
    print("Done! Labeled data with is_high_risk saved to data/processed/labeled_data.csv")

if __name__  == "main":
    main()