import os
import json
import hashlib
import aioredis
import logging

# === Redis Configuration ===
# Supports localhost, AWS ElastiCache or RedisCluster Proxy endpoints

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Try to connect to Redis safely
r = None
try:
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    # Test the connection
    import asyncio
    asyncio.get_event_loop().run_until_complete(r.ping())
except Exception as e:
    logging.warning(f"[Redis] Could not connect to Redis at {REDIS_URL}: {e}")
    r = None  # fallback: disable Redis usage


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def get_from_cache_async(hash_key: str):
    if r is None:
        return None
    try:
        result = await r.get(hash_key)
        return json.loads(result) if result else None
    except Exception as e:
        logging.warning(f"[Redis] Error getting key {hash_key}: {e}")
        return None


async def save_to_cache_async(hash_key: str, embedding: list):
    if r is None:
        return
    try:
        await r.set(hash_key, json.dumps(embedding), ex=60 * 60 * 24)
    except Exception as e:
        logging.warning(f"[Redis] Error saving key {hash_key}: {e}")