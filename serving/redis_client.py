import json
import logging
from typing import Optional

import redis.asyncio as redis

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger("redis")
logger.setLevel(logging.INFO)

redis_client: Optional[redis.Redis] = None


async def init_redis():
    global redis_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis connected (async)")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None


async def get_from_cache(key: str) -> Optional[dict]:
    if redis_client is None:
        return None

    value = await redis_client.get(key)
    if value:
        return json.loads(value)
    return None


async def set_to_cache(key: str, value: dict, ttl: int):
    if redis_client is None:
        return

    await redis_client.setex(key, ttl, json.dumps(value))
