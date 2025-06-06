"""
Common utility functions for AgenticRAG.
Provides shared functionality used across multiple modules.
"""

import logging
import os
import torch
from typing import Dict, Any, List, Optional, Union
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def get_device() -> str:
    """
    Get the device to use for model inference.

    Returns:
        str: The device to use ('cuda', 'mps', or 'cpu').
    """
    try:
        if torch.cuda.is_available():
            logger.info("Using CUDA for GPU acceleration.")
            return "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS for GPU acceleration.")
            return "mps"
        else:
            logger.info("Using CPU for processing.")
            return "cpu"
    except Exception as e:
        logger.warning(f"Error checking device availability: {str(e)}, falling back to CPU")
        return "cpu"

def get_text_splitter(chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> RecursiveCharacterTextSplitter:
    """
    Get a standardized text splitter for document chunking.
    Uses configuration values if not explicitly provided.

    Args:
        chunk_size: The size of each chunk (overrides config if provided)
        chunk_overlap: The overlap between chunks (overrides config if provided)

    Returns:
        RecursiveCharacterTextSplitter: The text splitter
    """
    # Import here to avoid circular imports
    from config import config

    # Use provided values or fall back to config
    chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    logger.debug(f"Creating text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def filter_complex_metadata(data: Any) -> Dict[str, Any]:
    """
    Filter out complex metadata that ChromaDB can't handle.

    Args:
        data: Either a Document object, metadata dictionary, or other type

    Returns:
        A filtered metadata dictionary with only simple types
    """
    try:
        # Handle Document objects (extract metadata)
        if hasattr(data, 'metadata') and data.metadata is not None:
            metadata = data.metadata
        # Handle dictionaries directly
        elif isinstance(data, dict):
            metadata = data
        # Handle case where input is a tuple or other non-dict type
        elif isinstance(data, tuple) or not hasattr(data, 'items'):
            # Silently return empty dict for tuples to avoid log spam
            return {}
        # Handle None
        elif data is None:
            return {}
        else:
            # Try to convert to dict if possible
            try:
                metadata = dict(data)
            except (TypeError, ValueError):
                # If conversion fails, return empty dict
                return {}

        # Now process the metadata dictionary
        filtered_metadata = {}
        for key, value in metadata.items():
            # Only keep simple types that ChromaDB can handle
            if value is None:
                # Replace None with empty string
                filtered_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                filtered_metadata[key] = value
            else:
                # Convert complex types to strings
                try:
                    filtered_metadata[key] = str(value)
                except Exception:
                    # If conversion fails, use a placeholder
                    filtered_metadata[key] = f"<complex value: {type(value).__name__}>"
        return filtered_metadata
    except Exception:
        # Return a safe default without printing to reduce noise
        return {"error": "Failed to process metadata"}

def clean_document_content(content: str) -> str:
    """
    Clean document content by removing null bytes and normalizing whitespace.

    Args:
        content: The document content to clean

    Returns:
        str: The cleaned content
    """
    if not isinstance(content, str):
        return ""

    # Remove null bytes
    cleaned = content.replace('\x00', '')

    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())

    # Ensure proper UTF-8 encoding
    cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')

    return cleaned.strip()

def validate_documents(documents: List[Document]) -> List[Document]:
    """
    Validate and clean a list of documents.

    Args:
        documents: The list of documents to validate

    Returns:
        List[Document]: The validated and cleaned documents
    """
    valid_documents = []

    for doc in documents:
        try:
            if not isinstance(doc, Document):
                logger.warning(f"Skipping non-Document object: {type(doc)}")
                continue

            if not hasattr(doc, 'page_content') or not doc.page_content:
                logger.warning("Skipping document with empty content")
                continue

            # Clean the content
            cleaned_content = clean_document_content(doc.page_content)

            if not cleaned_content:
                logger.warning("Skipping document with empty cleaned content")
                continue

            # Update the document with cleaned content
            doc.page_content = cleaned_content

            # Ensure metadata is properly filtered
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            else:
                doc.metadata = filter_complex_metadata(doc.metadata)

            valid_documents.append(doc)
        except Exception as e:
            logger.warning(f"Error validating document: {str(e)}")
            continue

    return valid_documents

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level: The log level to use
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress warnings and set environment variables
    suppress_warnings()

def suppress_warnings() -> None:
    """
    Suppress common warnings and set environment variables for better performance.
    This function centralizes all warning suppression in one place.
    """
    import warnings

    # Torch warnings
    warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes raised.*")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

    # HuggingFace tokenizer warnings
    warnings.filterwarnings("ignore", message=".*The current process just got forked.*")

    # Langchain warnings
    warnings.filterwarnings("ignore", message=".*The get_embeddings method is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Directly instantiating a Chroma.*")

    # Set environment variables for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # Set PyTorch environment variables if not already set
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count() or 4))

    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = str(min(8, os.cpu_count() or 4))
