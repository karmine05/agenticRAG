"""
Vector store management for AgenticRAG.
Provides a unified interface for vector store operations.
"""

import os
import logging
import tempfile
import shutil
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings

from config import config
from utils import get_device, get_text_splitter, validate_documents

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manager for vector store operations.
    Provides methods for creating, updating, and querying vector stores.
    """
    
    def __init__(self, persist_directory: str = None, embedding_model: str = None):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = persist_directory or config.db_dir
        self.embedding_model = embedding_model or config.embedding_model
        self.device = get_device()
        self.temp_db_dirs = []
        
        # Create the persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize the main vector store
        self.vectordb = self._initialize_vector_store()
        
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize the embeddings model.
        
        Returns:
            HuggingFaceEmbeddings: The initialized embeddings model
        """
        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            # Fall back to CPU if there's an error
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize the main vector store.
        
        Returns:
            Chroma: The initialized vector store
        """
        try:
            client_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
            
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the main vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return
            
        # Validate and clean the documents
        valid_documents = validate_documents(documents)
        
        if not valid_documents:
            logger.warning("No valid documents to add to vector store")
            return
            
        # Ensure documents are properly chunked
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents(valid_documents)
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Add the chunks to the vector store
        self.vectordb.add_documents(chunks)
        
    def create_temp_vector_store(self, documents: List[Document]) -> Tuple[Chroma, str]:
        """
        Create a temporary vector store for the session.
        
        Args:
            documents: List of documents to add to the temporary vector store
            
        Returns:
            Tuple[Chroma, str]: The temporary vector store and its directory
        """
        if not documents:
            logger.warning("No documents to add to temporary vector store")
            raise ValueError("No documents to add to temporary vector store")
            
        # Validate and clean the documents
        valid_documents = validate_documents(documents)
        
        if not valid_documents:
            logger.warning("No valid documents to add to temporary vector store")
            raise ValueError("No valid documents to add to temporary vector store")
            
        # Ensure documents are properly chunked
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents(valid_documents)
        
        logger.info(f"Creating temporary vector store with {len(chunks)} chunks")
        
        # Create a temporary directory
        temp_db_dir = tempfile.mkdtemp()
        self.temp_db_dirs.append(temp_db_dir)
        
        # Create the temporary vector store
        temp_vectordb = Chroma(
            persist_directory=temp_db_dir,
            embedding_function=self.embeddings,
            client_settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=False
            )
        )
        
        # Add the chunks to the temporary vector store
        temp_vectordb.add_documents(chunks)
        
        return temp_vectordb, temp_db_dir
        
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """
        Search the main vector store for documents similar to the query.
        
        Args:
            query: The query to search for
            k: Number of documents to retrieve
            
        Returns:
            List[Document]: The retrieved documents
        """
        logger.info(f"Searching vector store for query: {query}")
        return self.vectordb.similarity_search(query, k=k)
        
    def cleanup_temp_vector_stores(self) -> None:
        """
        Clean up temporary vector stores.
        """
        for temp_db_dir in self.temp_db_dirs:
            try:
                logger.info(f"Cleaning up temporary vector store: {temp_db_dir}")
                shutil.rmtree(temp_db_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error cleaning up temporary vector store: {str(e)}")
            
        # Clear the list of temporary directories
        self.temp_db_dirs = []
        
        # Force garbage collection
        try:
            import gc
            gc.collect()
        except ImportError:
            # Skip if gc is not available (during shutdown)
            pass
        
    def __del__(self):
        """
        Clean up resources when the manager is deleted.
        """
        try:
            self.cleanup_temp_vector_stores()
        except (ImportError, AttributeError, TypeError):
            # Ignore errors during interpreter shutdown
            pass

# Create a singleton instance
vector_store_manager = VectorStoreManager()
