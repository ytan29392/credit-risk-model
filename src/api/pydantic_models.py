from pydantic import BaseModel
import pandas as pd

class CustomerInput(BaseModel):
    age: int
    income: float
    transaction_count: int
    # Add all required model input features

    def to_df(self):
        return pd.DataFrame([self.dict()])

class RiskPrediction(BaseModel):
    probability: float
