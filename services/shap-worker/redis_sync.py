import redis
import logging
import os

logger = logging.getLogger("redis-sync")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))


redis_client = None


def init_redis_sync():
    global redis_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connected (sync)")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None


def is_processed(txn_id: str) -> bool:
    if redis_client is None:
        return False
    return redis_client.exists(f"processed:{txn_id}") == 1


def mark_processed(txn_id: str, ttl: int = 86400):
    if redis_client is None:
        return
    redis_client.setex(f"processed:{txn_id}", ttl, "1")