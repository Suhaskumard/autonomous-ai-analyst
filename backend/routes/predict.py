from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
import random
import time

router = APIRouter()

class PredictRequest(BaseModel):
    dataset_ref: str

@router.post("/predict")
async def predict_pipeline(request: PredictRequest) -> Dict:
    """
    Receive dataset reference, trigger processing, return mock model metrics
    """
    # Simulate processing time
    time.sleep(1.5)
    
    # Mocking machine learning prediction metrics
    accuracy = random.uniform(0.85, 0.98)
    rmse = random.uniform(0.1, 0.5)
    f1_score = random.uniform(0.82, 0.96)
    
    # Mock chart data
    chart_data = [
        {"name": "Jan", "actual": random.randint(30, 50), "predicted": random.randint(28, 52)},
        {"name": "Feb", "actual": random.randint(40, 60), "predicted": random.randint(38, 62)},
        {"name": "Mar", "actual": random.randint(50, 70), "predicted": random.randint(48, 72)},
        {"name": "Apr", "actual": random.randint(60, 80), "predicted": random.randint(58, 82)},
        {"name": "May", "actual": random.randint(70, 90), "predicted": random.randint(68, 92)},
    ]
    
    return {
        "status": "success",
        "dataset_ref": request.dataset_ref,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "rmse": round(rmse, 4),
            "f1_score": round(f1_score, 4)
        },
        "chart_data": chart_data,
        "message": "Model predictions generated successfully"
    }
