from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import streamlit as st

# Load the saved Logistic Regression model
model = joblib.load('logistic_regression_model.joblib')

# Initialize FastAPI app
app = FastAPI()

# Define the input data model (this will be the structure of the input data that you will send)
class InputData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Log the input data
        print(f"Received data: {data}")
        
        # Convert the input data into the format expected by the model (e.g., NumPy array)
        input_data = np.array([[data.feature_1, data.feature_2, data.feature_3, data.feature_4]])
        
        # Make the prediction using the model
        prediction = model.predict(input_data)
        
        # Log the prediction
        print(f"Prediction: {prediction}")
        
        # Return the prediction result
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


