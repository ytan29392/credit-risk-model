import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/processed/labeled_data.csv")
    df = df.dropna(subset=["is_high_risk"])
    X = df.drop(columns=["CustomerId", "is_high_risk", "TransactionStartTime", "TransactionId"])
    y = df["is_high_risk"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def run_experiments():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("credit-risk-model")

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, preds, probs)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.sklearn.log_model(model, model_name)

            print(f"{model_name} metrics:", metrics)

if __name__ == "__main__":
    run_experiments()