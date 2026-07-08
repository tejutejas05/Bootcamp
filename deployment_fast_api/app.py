from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib

app = FastAPI()

try:
    model = joblib.load("loan_model.pkl")
except Exception:
    model = None

class LoanRequest(BaseModel):
    Age: int = Field(..., ge=18,le=60,description="Age between 18 and 60")
    Salary: float = Field(..., gt=10000, description="Salary must be greater than 10000")

@app.get("/")
def home():
    return ({"message":"Welcome to the Loan Prediction API"})

@app.get("/health")
def health():
    
    if model is None:
        return {
            "status": "Unhealthy",
            "model": "Not Loaded"
        }
    return {
        "status": "Healthy",
        "model": "Loaded"
    }

@app.post("/predict")
def predict(data: LoanRequest):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model is not available"
        )
    
    if data.Salary > 100000:
        raise HTTPException(
         status_code=404,
         detail="Salary means unrealistic"   
        )

    try:

        input_data = [[data.Age, data.Salary]]
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            return {"message": "Loan Approved"}
        else:
            return {"message": "Loan Rejected"}

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Model cannot predict correctly."
        )
