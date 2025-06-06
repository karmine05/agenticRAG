"""
Cache manager for AgenticRAG.
Provides caching functionality for document retrieval and web search results.
"""

import os
import json
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
from langchain.schema import Document
from error_handler import handle_error, try_except_decorator, ErrorHandler

# Set up logging
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Cache manager for storing and retrieving results from previous queries.
    """

    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600, max_size_mb: int = 100):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            max_size_mb: Maximum cache size in megabytes (default: 100MB)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self._ensure_cache_dir()
        self._cleanup_expired_entries()

    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                logger.info(f"Created cache directory: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error creating cache directory: {str(e)}")

    def _get_cache_key(self, key_data: Any) -> str:
        """
        Generate a cache key from the provided data.
        Uses a more efficient approach for large inputs.

        Args:
            key_data: Data to generate a key from

        Returns:
            A string key
        """
        # For dictionaries, we'll hash each item separately to avoid large JSON serialization
        if isinstance(key_data, dict):
            hasher = hashlib.blake2b(digest_size=16)  # Faster than MD5

            # Sort keys for consistent hashing
            for key in sorted(key_data.keys()):
                # Add the key to the hash
                hasher.update(str(key).encode())

                # Add a separator
                hasher.update(b':')

                # Add the value to the hash
                value = key_data[key]
                if isinstance(value, (dict, list, tuple)):
                    # For complex types, use their hash
                    value_hash = self._get_cache_key(value)
                    hasher.update(value_hash.encode())
                else:
                    # For simple types, use their string representation
                    hasher.update(str(value).encode())

                # Add a separator between key-value pairs
                hasher.update(b',')

            return hasher.hexdigest()

        # For lists and tuples, hash each item separately
        elif isinstance(key_data, (list, tuple)):
            hasher = hashlib.blake2b(digest_size=16)

            for item in key_data:
                if isinstance(item, (dict, list, tuple)):
                    # For complex types, use their hash
                    item_hash = self._get_cache_key(item)
                    hasher.update(item_hash.encode())
                else:
                    # For simple types, use their string representation
                    hasher.update(str(item).encode())

                # Add a separator between items
                hasher.update(b',')

            return hasher.hexdigest()

        # For simple types, use their string representation
        else:
            key_str = str(key_data)
            return hashlib.blake2b(key_str.encode(), digest_size=16).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.

        Args:
            cache_key: The cache key

        Returns:
            The file path
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def get(self, key_data: Any) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key_data: Data to generate a key from

        Returns:
            The cached value, or None if not found or expired
        """
        try:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)

            # Check if the cache file exists
            if not os.path.exists(cache_path):
                return None

            # Read the cache file
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Get the TTL for this entry (use custom TTL if available, otherwise use default)
            entry_ttl = cache_data.get('ttl', self.ttl)

            # Check if the cache entry has expired
            if time.time() - cache_data['timestamp'] > entry_ttl:
                logger.info(f"Cache entry expired: {cache_key}")
                # Try to remove the expired file
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
                return None

            logger.info(f"Cache hit: {cache_key}")
            return cache_data['value']

        except Exception as e:
            logger.error(f"Error getting cache entry: {str(e)}")
            return None

    def set(self, key_data: Any, value: Any) -> bool:
        """
        Set a value in the cache.

        Args:
            key_data: Data to generate a key from
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)

            # Prepare the cache data
            cache_data = {
                'timestamp': time.time(),
                'value': value
            }

            # Write the cache file
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)

            logger.info(f"Cache set: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error setting cache entry: {str(e)}")
            return False

    def invalidate(self, key_data: Any) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key_data: Data to generate a key from

        Returns:
            True if successful, False otherwise
        """
        try:
            cache_key = self._get_cache_key(key_data)
            cache_path = self._get_cache_path(cache_key)

            # Check if the cache file exists
            if not os.path.exists(cache_path):
                return True

            # Remove the cache file
            os.remove(cache_path)
            logger.info(f"Cache invalidated: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error invalidating cache entry: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove all files in the cache directory
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            logger.info("Cache cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

    def _cleanup_expired_entries(self) -> None:
        """
        Clean up expired cache entries and enforce size limits.
        """
        try:
            # Get all cache files
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Get file stats
                        stats = os.stat(file_path)
                        size = stats.st_size

                        # Try to read the file to get timestamp
                        with open(file_path, 'r') as f:
                            cache_data = json.load(f)
                            timestamp = cache_data.get('timestamp', 0)

                        # Add to list of cache files
                        cache_files.append({
                            'path': file_path,
                            'size': size,
                            'timestamp': timestamp,
                            'is_expired': (time.time() - timestamp) > self.ttl
                        })
                    except Exception:
                        # If we can't read the file, consider it expired
                        cache_files.append({
                            'path': file_path,
                            'size': os.path.getsize(file_path),
                            'timestamp': 0,
                            'is_expired': True
                        })

            # First, remove expired entries
            for file_info in cache_files:
                if file_info['is_expired']:
                    try:
                        os.remove(file_info['path'])
                        logger.debug(f"Removed expired cache entry: {file_info['path']}")
                    except Exception as e:
                        logger.warning(f"Failed to remove expired cache entry: {str(e)}")

            # Then, check if we're still over the size limit
            remaining_files = [f for f in cache_files if not f['is_expired']]
            total_size = sum(f['size'] for f in remaining_files)

            if total_size > self.max_size_bytes:
                # Sort by timestamp (oldest first)
                remaining_files.sort(key=lambda x: x['timestamp'])

                # Remove oldest files until we're under the limit
                for file_info in remaining_files:
                    if total_size <= self.max_size_bytes:
                        break

                    try:
                        os.remove(file_info['path'])
                        total_size -= file_info['size']
                        logger.debug(f"Removed old cache entry to enforce size limit: {file_info['path']}")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache entry: {str(e)}")

            logger.info(f"Cache cleanup complete. Current size: {total_size / (1024 * 1024):.2f} MB")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Count the number of cache files
            cache_files = [f for f in os.listdir(self.cache_dir) if os.path.isfile(os.path.join(self.cache_dir, f))]
            cache_count = len(cache_files)

            # Calculate the total size of the cache
            cache_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)

            # Count the number of expired cache entries
            expired_count = 0
            for filename in cache_files:
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)

                    if time.time() - cache_data['timestamp'] > self.ttl:
                        expired_count += 1
                except:
                    # Skip files that can't be read
                    pass

            return {
                'count': cache_count,
                'size': cache_size,
                'size_mb': cache_size / (1024 * 1024),
                'expired': expired_count,
                'ttl': self.ttl,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'directory': self.cache_dir
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                'error': str(e)
            }

# Create a singleton instance
cache_manager = CacheManager()

def cached(ttl: Optional[int] = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time-to-live for cache entries in seconds (default: use cache manager's TTL)

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function name, args, and kwargs
            cache_key = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }

            # Check if the result is in the cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call the function
            result = func(*args, **kwargs)

            # Store the result in the cache with custom TTL if provided
            if ttl is not None:
                # Create a custom cache entry with the specified TTL
                cache_path = cache_manager._get_cache_path(cache_manager._get_cache_key(cache_key))

                # Prepare the cache data with custom TTL
                cache_data = {
                    'timestamp': time.time(),
                    'value': result,
                    'ttl': ttl  # Store custom TTL for this entry
                }

                # Write the cache file
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache_data, f)
                    logger.info(f"Cache set with custom TTL of {ttl}s: {func.__name__}")
                except Exception as e:
                    logger.error(f"Error setting cache entry with custom TTL: {str(e)}")
            else:
                # Use the default cache manager TTL
                cache_manager.set(cache_key, result)

            return result
        return wrapper
    return decorator

def cached_document_retrieval(func):
    """
    Decorator for caching document retrieval results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(query: str, vectordb, k: int = 8):
        # Create a cache key from the query and k
        cache_key = {
            'function': func.__name__,
            'query': query,
            'k': k
        }

        # Check if the result is in the cache
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            # Convert the cached result back to Document objects
            documents = []
            for doc_dict in cached_result:
                doc = Document(
                    page_content=doc_dict['page_content'],
                    metadata=doc_dict['metadata']
                )
                documents.append(doc)
            return documents

        # Call the function
        documents = func(query, vectordb, k)

        # Convert Document objects to dictionaries for caching
        doc_dicts = []
        for doc in documents:
            doc_dict = {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            doc_dicts.append(doc_dict)

        # Store the result in the cache
        cache_manager.set(cache_key, doc_dicts)

        return documents
    return wrapper

def cached_web_search(func):
    """
    Decorator for caching web search results.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(query: str, *args, **kwargs):
        # Create a cache key from the query
        cache_key = {
            'function': func.__name__,
            'query': query,
            'args': args,
            'kwargs': kwargs
        }

        # Check if the result is in the cache
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Call the function
        result = func(query, *args, **kwargs)

        # Store the result in the cache
        cache_manager.set(cache_key, result)

        return result
    return wrapper
