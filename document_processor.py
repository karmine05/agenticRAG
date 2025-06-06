"""
Document processor for AgenticRAG.
Provides unified functions for processing documents from various sources.
"""

import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import time
from utils.common import get_text_splitter, get_device, clean_document_content, filter_complex_metadata
from error_handler import handle_error, try_except_decorator, ErrorHandler, DocumentProcessingError, VectorStoreError, WebScrapingError
from parallel_utils import parallel_document_processing

# Set up logging
logger = logging.getLogger(__name__)

def process_pdf(file_path: str) -> List[Document]:
    """
    Process a PDF file and return document chunks.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of document chunks
    """
    logger.info(f"Processing PDF file: {file_path}")

    try:
        # Read PDF
        pdf_reader = PdfReader(file_path)
        text = ""

        # Extract text from each page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add non-empty text
                text += page_text + "\\n"

        # Check if any text was extracted
        if not text.strip():
            logger.warning(f"No text content could be extracted from {file_path}")
            return []

        # Use standardized text splitter
        text_splitter = get_text_splitter()

        # Create Document objects with proper metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": os.path.basename(file_path),
                "type": "pdf",
                "path": file_path
            }
        )

        # Split the document into chunks
        chunks = text_splitter.split_documents([doc])

        logger.info(f"Successfully extracted {len(chunks)} chunks from PDF")
        return chunks

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        return []

def process_pdfs_in_parallel(file_paths: List[str], max_workers: Optional[int] = None) -> List[Document]:
    """
    Process multiple PDF files in parallel and return document chunks.

    Args:
        file_paths: List of paths to PDF files
        max_workers: Maximum number of worker threads

    Returns:
        List of document chunks from all PDFs
    """
    logger.info(f"Processing {len(file_paths)} PDF files in parallel")

    # Use parallel processing to process PDFs
    all_chunks = parallel_document_processing(process_pdf, file_paths, max_workers=max_workers)

    logger.info(f"Successfully processed {len(file_paths)} PDFs with {len(all_chunks)} total chunks")
    return all_chunks

def process_website(url: str) -> List[Document]:
    """
    Process a website and return document chunks.

    Args:
        url: URL of the website

    Returns:
        List of document chunks
    """
    logger.info(f"Processing website: {url}")

    try:
        # Fetch and parse website content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        logger.info(f"Successfully fetched website with status code: {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content (excluding scripts, styles, and other non-content elements)
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            element.decompose()

        # Get the main content with priority to important sections
        main_content = None
        content_priority = [
            ('main', None),
            ('article', None),
            ('div', {'class_': 'content'}),
            ('div', {'class_': 'main-content'}),
            ('div', {'class_': 'article-content'}),
            ('div', {'id': 'content'}),
            ('div', {'id': 'main-content'})
        ]

        for tag, attrs in content_priority:
            if attrs:
                main_content = soup.find(tag, attrs)
            else:
                main_content = soup.find(tag)
            if main_content:
                break

        # If no main content found, try to find the largest text container
        if not main_content:
            text_containers = soup.find_all(['div', 'section', 'article'])
            if text_containers:
                main_content = max(text_containers, key=lambda x: len(x.get_text()))

        # Extract text from the main content or the whole page
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()

        # Clean the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Use standardized text splitter
        text_splitter = get_text_splitter()

        # Create Document object with metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "type": "website",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "title": soup.title.string if soup.title else "Untitled"
            }
        )

        # Split the document into chunks
        chunks = text_splitter.split_documents([doc])

        logger.info(f"Successfully extracted {len(chunks)} chunks from website")
        return chunks

    except Exception as e:
        logger.error(f"Error processing website {url}: {str(e)}")
        return []

def process_websites_in_parallel(urls: List[str], max_workers: Optional[int] = None) -> List[Document]:
    """
    Process multiple websites in parallel and return document chunks.

    Args:
        urls: List of website URLs
        max_workers: Maximum number of worker threads

    Returns:
        List of document chunks from all websites
    """
    logger.info(f"Processing {len(urls)} websites in parallel")

    # Use parallel processing to process websites
    all_chunks = parallel_document_processing(process_website, urls, max_workers=max_workers)

    logger.info(f"Successfully processed {len(urls)} websites with {len(all_chunks)} total chunks")
    return all_chunks

def process_document_content(documents: List[Document]) -> List[Document]:
    """
    Process and validate document content using utility functions.

    Args:
        documents: List of documents to clean

    Returns:
        List of cleaned documents
    """
    valid_documents = []

    for doc in documents:
        try:
            if isinstance(doc.page_content, str):
                # Clean content using the utility function
                cleaned_content = clean_document_content(doc.page_content)

                # Check if content is not empty after cleaning
                if cleaned_content:
                    doc.page_content = cleaned_content

                    # Filter metadata using the utility function
                    if hasattr(doc, 'metadata') and doc.metadata is not None:
                        doc.metadata = filter_complex_metadata(doc.metadata)
                    else:
                        doc.metadata = {}

                    valid_documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing document content: {str(e)}")
            continue

    return valid_documents

def create_vector_store(documents: List[Document], persist_directory: str) -> Chroma:
    """
    Create or update a vector store with the given documents.

    Args:
        documents: List of documents to add to the vector store
        persist_directory: Directory to persist the vector store

    Returns:
        Chroma vector store
    """
    # Clean and validate documents
    valid_documents = process_document_content(documents)

    if not valid_documents:
        raise ValueError("No valid documents to process after filtering")

    # Ensure documents are properly chunked
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(valid_documents)

    # Filter metadata for each chunk
    for chunk in chunks:
        chunk.metadata = filter_complex_metadata(chunk.metadata)

    # Get the device
    device = get_device()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create or update vector store
    try:
        # Import ChromaDB settings
        import chromadb
        from chromadb.config import Settings

        # Create proper Settings object
        client_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

        # Create vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            client_settings=client_settings
        )

        return vectordb

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def create_temp_vector_store(documents: List[Document]) -> Tuple[Chroma, str]:
    """
    Create a temporary vector store for the session.

    Args:
        documents: List of documents to add to the vector store

    Returns:
        Tuple of (Chroma vector store, temporary directory path)
    """
    # Clean and validate documents
    valid_documents = process_document_content(documents)

    if not valid_documents:
        raise ValueError("No valid documents to process after filtering")

    # Ensure documents are properly chunked
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(valid_documents)

    # Filter metadata for each chunk
    for chunk in chunks:
        chunk.metadata = filter_complex_metadata(chunk.metadata)

    # Get the device
    device = get_device()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create temporary directory
    temp_db_dir = tempfile.mkdtemp()

    # Create vector store
    try:
        # Import ChromaDB settings
        import chromadb
        from chromadb.config import Settings

        # Create proper Settings object
        client_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False
        )

        # Create vector store
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=temp_db_dir,
            client_settings=client_settings
        )

        return vectordb, temp_db_dir

    except Exception as e:
        logger.error(f"Error creating temporary vector store: {str(e)}")
        raise

def cleanup_vector_store(temp_db_dir: str) -> bool:
    """
    Clean up a temporary vector store.

    Args:
        temp_db_dir: Path to the temporary directory

    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        import shutil
        import time

        # Add a small delay to ensure all file handles are released
        time.sleep(0.5)

        # Remove the directory
        if os.path.exists(temp_db_dir):
            shutil.rmtree(temp_db_dir, ignore_errors=True)

        return True

    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_db_dir}: {str(e)}")
        return False
