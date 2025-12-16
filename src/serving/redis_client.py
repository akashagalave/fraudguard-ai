import json
import redis
import logging
from typing import Optional

from .config import REDIS_HOST, REDIS_PORT, REDIS_DB

logger = logging.getLogger("redis")
logger.setLevel(logging.INFO)

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_client = None


def get_from_cache(key: str) -> Optional[dict]:
    if redis_client is None:
        return None

    value = redis_client.get(key)
    if value:
        logger.info("CACHE HIT")
        return json.loads(value)

    logger.info("CACHE MISS")
    return None


def set_to_cache(key: str, value: dict, ttl: int):
    if redis_client is None:
        return

    redis_client.setex(key, ttl, json.dumps(value))
