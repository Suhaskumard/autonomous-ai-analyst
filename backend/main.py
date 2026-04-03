from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routes import upload, predict, insights
from utils.helpers import prepare_backend_dirs

# Ensure directories exist
prepare_backend_dirs()

app = FastAPI(
    title="Autonomous AI Analyst API",
    description="Backend API for full-stack AI data analysis system",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URLs instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload.router, tags=["Upload"])
app.include_router(predict.router, tags=["Predict"])
app.include_router(insights.router, tags=["Insights"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Autonomous AI Analyst API. Visit /docs for documentation."}
