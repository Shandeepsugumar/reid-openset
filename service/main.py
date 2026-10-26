from fastapi import FastAPI

app = FastAPI(
    title="Open-Set Person ReID Service",
    description="Inference and identity verification service with open-set rejection.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "reid-openset"}
