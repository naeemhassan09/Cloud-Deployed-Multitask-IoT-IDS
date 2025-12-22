# api/main.py
from fastapi import FastAPI

app = FastAPI(title="Multitask IoT IDS API")

@app.get("/health")
async def health():
    return {"status": "ok"}