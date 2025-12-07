import redis
import json
import hashlib
from typing import Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Cache:
    """Redis cache for ML inference results"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis connection
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.redis_client = None
        self.enabled = False
        self.host = host
        self.port = port
        self.db = db
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"✓ Redis cache connected: {host}:{port}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis not available: {e}. Running without cache.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.enabled = False
    
    def generate_key(self, file_path: Path, operation: str) -> str:
        """
        Generate cache key from file content and operation
        
        Args:
            file_path: Path to file
            operation: Operation type (classify, extract, analyze)
            
        Returns:
            str: Cache key
        """
        # Hash file content
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        
        # Create key: operation:file_hash
        return f"{operation}:{file_hash}"
    
    def get(self, key: str) -> Optional[dict]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            dict or None: Cached value
        """
        if not self.enabled:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                logger.info(f"✓ Cache HIT: {key[:20]}...")
                return json.loads(value)
            logger.info(f"✗ Cache MISS: {key[:20]}...")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        try:
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.info(f"✓ Cached: {key[:20]}... (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled:
            return False
        
        try:
            self.redis_client.delete(key)
            logger.info(f"✓ Cache deleted: {key[:20]}...")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache (use with caution!)"""
        if not self.enabled:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("⚠️  Cache cleared!")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.redis_client.info()
            return {
                "enabled": True,
                "keys": self.redis_client.dbsize(),
                "memory_used": info.get("used_memory_human", "N/A"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": False, "error": str(e)}
    
    @staticmethod
    def _calculate_hit_rate(hits: int, misses: int) -> str:
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return "N/A"
        rate = (hits / total) * 100
        return f"{rate:.2f}%"

# Global cache instance
cache = Cache()