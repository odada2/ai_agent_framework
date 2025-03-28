# ai_agent_framework/core/state/redis_manager.py

import asyncio
import logging
import pickle
from typing import Optional, Any

import redis.asyncio as redis # Import async redis client

# Assuming ConversationMemory can be imported
from ai_agent_framework.core.memory.conversation import ConversationMemory
# Assuming Settings can be imported
from ai_agent_framework.config.settings import Settings

logger = logging.getLogger(__name__)

class RedisStateManager:
    """
    Manages agent state (specifically ConversationMemory) persistence using Redis.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._redis_client: Optional[redis.Redis] = None
        self._prefix = "agent_memory:"
        self._ttl = self.settings.get("storage.redis.session_ttl", 3600) # Default TTL

    async def _get_client(self) -> redis.Redis:
        """Initializes and returns the async Redis client."""
        if self._redis_client is None:
            host = self.settings.get("storage.redis.host", "localhost")
            port = self.settings.get("storage.redis.port", 6379)
            db = self.settings.get("storage.redis.db", 0)
            password = self.settings.get("storage.redis.password", None)

            try:
                logger.info(f"Connecting to Redis at {host}:{port} DB {db}")
                # Use redis.asyncio.from_url for simpler setup if preferred
                self._redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False # Store bytes (for pickle)
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis connection successful.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
                self._redis_client = None # Ensure client is None on failure
                raise ConnectionError(f"Could not connect to Redis: {e}")
        return self._redis_client

    def _get_redis_key(self, conversation_id: str) -> str:
        """Generates the Redis key for a given conversation ID."""
        return f"{self._prefix}{conversation_id}"

    async def save_memory(self, conversation_id: str, memory: ConversationMemory):
        """
        Saves the ConversationMemory object to Redis using pickle.

        Args:
            conversation_id: The unique ID for the conversation.
            memory: The ConversationMemory object to save.
        """
        if not isinstance(memory, ConversationMemory):
            logger.error(f"Attempted to save non-ConversationMemory object for {conversation_id}")
            return

        try:
            client = await self._get_client()
            if not client: return # Cannot save if client failed to init

            redis_key = self._get_redis_key(conversation_id)
            serialized_memory = pickle.dumps(memory)

            # Save to Redis with TTL
            await client.setex(redis_key, self._ttl, serialized_memory)
            logger.debug(f"Saved memory for conversation {conversation_id} to Redis (TTL: {self._ttl}s)")

        except pickle.PicklingError as e:
            logger.error(f"Failed to serialize memory for conversation {conversation_id}: {e}", exc_info=True)
        except redis.RedisError as e:
            logger.error(f"Redis error saving memory for conversation {conversation_id}: {e}", exc_info=True)
        except Exception as e:
            logger.exception(f"Unexpected error saving memory for conversation {conversation_id}: {e}")


    async def load_memory(self, conversation_id: str) -> Optional[ConversationMemory]:
        """
        Loads a ConversationMemory object from Redis.

        Args:
            conversation_id: The unique ID for the conversation.

        Returns:
            The loaded ConversationMemory object or None if not found or error occurred.
        """
        try:
            client = await self._get_client()
            if not client: return None

            redis_key = self._get_redis_key(conversation_id)
            serialized_memory = await client.get(redis_key)

            if serialized_memory:
                try:
                    memory = pickle.loads(serialized_memory)
                    if isinstance(memory, ConversationMemory):
                        logger.debug(f"Loaded memory for conversation {conversation_id} from Redis.")
                        # Optionally refresh TTL on load
                        # await client.expire(redis_key, self._ttl)
                        return memory
                    else:
                        logger.error(f"Deserialized object for {conversation_id} is not ConversationMemory.")
                        # Optionally delete invalid key
                        # await client.delete(redis_key)
                        return None
                except pickle.UnpicklingError as e:
                    logger.error(f"Failed to deserialize memory for conversation {conversation_id}: {e}. Data might be corrupted.", exc_info=True)
                     # Optionally delete invalid key
                    # await client.delete(redis_key)
                    return None
            else:
                logger.debug(f"No memory found in Redis for conversation {conversation_id}.")
                return None # Not found

        except redis.RedisError as e:
            logger.error(f"Redis error loading memory for conversation {conversation_id}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.exception(f"Unexpected error loading memory for conversation {conversation_id}: {e}")
            return None

    async def delete_memory(self, conversation_id: str) -> bool:
        """
        Deletes conversation memory from Redis.

        Args:
            conversation_id: The ID of the conversation memory to delete.

        Returns:
            True if deletion was successful or key didn't exist, False otherwise.
        """
        try:
            client = await self._get_client()
            if not client: return False

            redis_key = self._get_redis_key(conversation_id)
            result = await client.delete(redis_key)
            logger.debug(f"Deleted memory for conversation {conversation_id} from Redis (Result: {result})")
            return True # Returns number of keys deleted (0 or 1)
        except redis.RedisError as e:
            logger.error(f"Redis error deleting memory for conversation {conversation_id}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error deleting memory for conversation {conversation_id}: {e}")
            return False

    async def close(self):
        """Closes the Redis connection."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                logger.info("Redis connection closed.")
            except redis.RedisError as e:
                 logger.error(f"Error closing Redis connection: {e}", exc_info=True)
            finally:
                self._redis_client = None