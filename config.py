"""
Configuration module for AgenticRAG.
Provides centralized configuration management for all components.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""

    def __init__(self):
        """Initialize the configuration."""
        # Model settings
        self.reasoning_model_id = os.getenv("REASONING_MODEL_ID", "llama3")
        self.tool_model_id = os.getenv("TOOL_MODEL_ID", "llama3")
        self.ioc_extraction_mode = os.getenv("IOC_EXTRACTION_MODE", "false").lower() in ["true", "yes", "1"]
        self.ioc_tool_model_id = os.getenv("IOC_TOOL_MODEL_ID", "qwen2.5:7b-64k")
        self.general_tool_model_id = os.getenv("GENERAL_TOOL_MODEL_ID", "llama3:8b-64k")
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")

        # Vector store settings
        self.db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.temp_db_boost = float(os.getenv("TEMP_DB_BOOST", "2.0"))

        # Text splitter settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "2900"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))

        # Web search settings
        self.web_search_enabled = os.getenv("WEB_SEARCH_ENABLED", "false").lower() in ["true", "yes", "1"]
        self.search_provider = os.getenv("SEARCH_PROVIDER", "serpapi").lower()
        self.serpapi_key = os.getenv("SERPAPI_KEY", "")

        # Device settings
        self.use_huggingface = os.getenv("USE_HUGGINGFACE", "no").lower() in ["yes", "true", "1"]

        # Logging settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Data settings
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # Check required settings
        if not self.reasoning_model_id:
            print("Error: REASONING_MODEL_ID is not set")
            return False

        if not self.tool_model_id:
            print("Error: TOOL_MODEL_ID is not set")
            return False

        # Check web search settings
        if self.web_search_enabled and self.search_provider == "serpapi" and not self.serpapi_key:
            print("Warning: Web search is enabled with SerpAPI but no API key is provided")

        # Create directories if they don't exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        return True

    def get_device(self) -> str:
        """
        Get the device to use for model inference.

        Returns:
            str: The device to use ('cuda', 'mps', or 'cpu').
        """
        import torch

        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except Exception:
            return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dict[str, Any]: The configuration as a dictionary.
        """
        return {
            "reasoning_model_id": self.reasoning_model_id,
            "tool_model_id": self.tool_model_id,
            "ioc_extraction_mode": self.ioc_extraction_mode,
            "ioc_tool_model_id": self.ioc_tool_model_id,
            "general_tool_model_id": self.general_tool_model_id,
            "db_dir": self.db_dir,
            "embedding_model": self.embedding_model,
            "temp_db_boost": self.temp_db_boost,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "web_search_enabled": self.web_search_enabled,
            "search_provider": self.search_provider,
            "use_huggingface": self.use_huggingface,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "device": self.get_device()
        }

# Create a singleton instance
config = Config()

# Validate the configuration
config.validate()
