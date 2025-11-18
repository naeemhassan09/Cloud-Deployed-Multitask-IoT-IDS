from fastapi import FastAPI

app = FastAPI(title="IoT Multitask IDS API")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: add /predict endpoint with Pydantic models and model inference
