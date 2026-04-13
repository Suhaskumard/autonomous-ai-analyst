from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routes.chat import router as chat_router
from routes.insights import router as insights_router
from routes.predict import router as predict_router
from routes.upload import router as upload_router

app = FastAPI(title="Autonomous AI Analyst", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(predict_router, prefix="/api", tags=["predict"])
app.include_router(insights_router, prefix="/api", tags=["insights"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/")
def health_check() -> dict:
    return {"message": "AUTONOMOUS AI ANALYST is running"}
