import os
import time
import asyncio
from typing import List
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_pipeline import pipeline_run
from fastapi.security.api_key import APIKeyHeader, APIKey
from prometheus_fastapi_instrumentator import Instrumentator

from fastapi import FastAPI, HTTPException, Header, Depends, Security

# --- Constants ---
API_SECRET_KEY = "Enter_Your_Secret_Key"
API_KEY_NAME = "X-API-Key"

# --- FastAPI App ---
app = FastAPI(
    title="LangChain Embedding API",
    description="Runs embedding + caching + FAISS storage via pipeline",
    version="1.0.0"
)

# --- Prometheus Metrics ---
instrumentator = Instrumentator().instrument(app).expose(app)

# --- Security Scheme (for docs) ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- Dependency to Check API Key ---
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return api_key

# --- Request Model ---
class EmbedRequest(BaseModel):
    texts: List[str]
    use_cache: bool = True
    store_vectors: bool = False

# --- POST Endpoint ---
@app.post("/pipeline/embed")
def run_pipeline(
    req: EmbedRequest,
    api_key: APIKey = Depends(verify_api_key)
):
    start_time = time.time()
    try:
        result = asyncio.run(pipeline_run(req.texts, req.use_cache, req.store_vectors))
        elapsed = time.time() - start_time
        result["time_taken_seconds"] = round(elapsed, 3)  # add timing to response
        print(f"[INFO] Pipeline run took {elapsed:.3f} seconds")  # <-- LOGGING LINE
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
