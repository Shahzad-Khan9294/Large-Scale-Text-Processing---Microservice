import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from vector_store import create_vectorstore 
from embeddings import SnowflakeEmbeddingModel
from utils import compute_hash, get_from_cache_async, save_to_cache_async

app = FastAPI(
    title="Micro_Services_Embeddings",
    description="Generates Sentence Embeddings Using Snowflake's Arctic Embed Model",
    version="1.0.0"
)

# Load model once
embedder = SnowflakeEmbeddingModel()

# Request & Response Models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    save_to_vectorstore: bool = False  

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    cache_stats: dict
    vectorstore_info: dict = None   

@app.get("/")
async def health_check():
    return {"status": "up", "message": "Embedding service is running!"}

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        hits, misses = 0, 0
        final_embeddings = []
        texts_to_compute = []
        compute_indices = []
        hash_keys = []

        for idx, text in enumerate(request.texts):
            h = compute_hash(text)
            cached = await get_from_cache_async(h)
            if cached:
                hits += 1
                final_embeddings.append(cached)
            else:
                misses += 1
                texts_to_compute.append(text)
                compute_indices.append((idx, h))
                final_embeddings.append(None)
            hash_keys.append(h)

        if texts_to_compute:
            computed = embedder.embed(texts_to_compute)
            for i, (original_index, hash_key) in enumerate(compute_indices):
                emb = computed[i]
                await save_to_cache_async(hash_key, emb)
                final_embeddings[original_index] = emb

        # ✅ Optional: Save to FAISS vector store
        vectorstore_info = None
        if request.save_to_vectorstore:
            # Only text chunks, not hashes
            vectorstore = create_vectorstore(request.texts, embeddings=embedder)
            save_path = "faiss_index"
            vectorstore.save_local(save_path)
            vectorstore_info = {
                "saved": True,
                "path": save_path,
                "num_vectors": len(request.texts)
            }

        return {
            "embeddings": final_embeddings,
            "cache_stats": {"hits": hits, "misses": misses},
            "vectorstore_info": vectorstore_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
