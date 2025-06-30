import pandas as pd
from scripts.proxy_target import calculate_rfm

def test_rfm_calculation():
    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B", "B"],
        "TransactionId": [1, 2, 3, 4],
        "TransactionStartTime": ["2024-01-01", "2024-01-05", "2024-01-01", "2024-01-02"],
        "Amount": [100, 200, 50, 150]
    })
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    rfm = calculate_rfm(df)
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns