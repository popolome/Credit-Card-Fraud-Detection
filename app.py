from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# CORS middleware setup to allow communication between frontend and backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load only the Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Feature columns expected by the model
feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

class TransactionData(BaseModel):
    V1: float
    Amount: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Credit Card Fraud Detection API!"}

@app.post("/predict/")
async def predict(transaction: TransactionData):
    # Prepare input vector for prediction
    # Time = 0, V1 from input, V2-V28 = 0, Amount from input
    input_vector = [0] + [transaction.V1] + [0]*27 + [transaction.Amount]

    # Convert input to a DataFrame with the correct feature columns
    input_df = pd.DataFrame([input_vector], columns=feature_columns)

    # Make prediction using the Random Forest model
    prediction_rf = rf_model.predict(input_df)[0]

    # Return the result as a JSON object
    return {
        "Random Forest Prediction": int(prediction_rf)  # 1 for Fraud, 0 for Non-Fraud
    }

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
