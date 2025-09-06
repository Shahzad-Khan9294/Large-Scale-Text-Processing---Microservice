from embeddings import SnowflakeEmbeddingModel

embedder = SnowflakeEmbeddingModel()

async def pipeline_run(texts: list[str], use_cache: bool = True, store_vectors: bool = False) -> dict:
    # ignoring use_cache and store_vectors, just embed directly
    computed = embedder.embed(texts)

    return {
        "embeddings": computed,
        "cache_stats": {"hits": 0, "misses": len(texts)},
        "vectorstore_info": None
    }
