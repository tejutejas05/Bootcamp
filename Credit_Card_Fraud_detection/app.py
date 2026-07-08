from fastapi import FastAPI
from pydantic import BaseModel,Field
import  joblib

app = FastAPI()

try:
    model = joblib.load("creditcard_model.pkl")
except Exception:
    model = None

class CreditCardRequest(BaseModel):
    v26: float = Field(..., description="Feature v26")
    v4: float = Field(..., description="Feature v4")
    v18: float = Field(..., description="Feature v18")
    v9: float = Field(..., description="Feature v9")
    v16: float = Field(..., description="Feature v16")
    v11: float = Field(..., description="Feature v11")
    v10: float = Field(..., description="Feature v10")
    v17: float = Field(..., description="Feature v17")
    v14: float = Field(..., description="Feature v14")
    v12: float = Field(..., description="Feature v12")


@app.get("/")
def home():
    return {
        "message" : "Welcome to Credit Card Fraud Detection API"
    }

@app.get("/health")
def health():
    if model is None:
        return {
            "status" : "Unhealthy",
            "model" : "not loaded"
        }
    return {
        "status" : "Healthy",
        "model" : "loaded"
    }

@app.post("/predict")
def predict(request : CreditCardRequest):

    if model is None:
        raise HTTPException(
            status_code = 400,
            detail = "internal server error"
        )
    
    try:

        features = [[
            request.v26,
            request.v4,
            request.v18,
            request.v9,
            request.v16,
            request.v11,
            request.v10,
            request.v17,
            request.v14,
            request.v12,
        ]]

        prediction = model.predict(features)

        if prediction[0] == 1:
            return {"message": "Credit Card Fraud Detected"}
        else:
            return {"message": "Credit Card Fraud Not Detected"}
    
    except Exception:
        raise HTTPException(
            status_code = 400,
            detail = "Prediction could not be performed"
        )
