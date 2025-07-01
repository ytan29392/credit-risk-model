from fastapi import FastAPI
from pydantic_models import CustomerInput, RiskPrediction
import mlflow.pyfunc

app = FastAPI()

# Load model from MLflow Registry
model = mlflow.pyfunc.load_model("models:/best_credit_model/Production")

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(data: CustomerInput):
    input_df = data.to_df()
    prediction = model.predict(input_df)
    return RiskPrediction(probability=float(prediction[0]))
