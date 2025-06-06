"""
Threat this script as a Admin panel for AgenticRAG. 
Run this script if you want to add new documents to the permanet vector database.
Processes documents from various sources and adds them to the vector database.
"""

import os
import logging
import glob
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any
from tqdm import tqdm
import requests
import time
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.schema import Document
from dotenv import load_dotenv

from config import config
from utils import (
    setup_logging,
    get_text_splitter,
    filter_complex_metadata,
    clean_document_content,
    validate_documents
)
from vector_store import vector_store_manager

# Set up logging
setup_logging(config.log_level)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_processed_files(persist_directory: str) -> Set[str]:
    """
    Get set of already processed files from tracking file.
    
    Args:
        persist_directory: Directory where the tracking file is stored
        
    Returns:
        Set of processed file paths
    """
    tracking_file = os.path.join(persist_directory, "processed_files.txt")
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def update_processed_files(persist_directory: str, new_files: Set[str]) -> None:
    """
    Update the tracking file with newly processed files.
    
    Args:
        persist_directory: Directory where the tracking file is stored
        new_files: Set of new file paths to add to the tracking file
    """
    tracking_file = os.path.join(persist_directory, "processed_files.txt")
    with open(tracking_file, 'a') as f:
        for file_path in new_files:
            f.write(f"{file_path}\n")

def process_single_pdf(file_path: str) -> List[Document]:
    """
    Process a single PDF file and return document chunks.
    
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
                text += page_text + "\n"
                
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
        
        # Validate and clean the chunks
        valid_chunks = validate_documents(chunks)
        
        logger.info(f"Successfully extracted {len(valid_chunks)} chunks from PDF")
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def process_single_file(file_path: str) -> List[Document]:
    """
    Process a single text or JSON file and return document chunks.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of document chunks
    """
    file_extension = Path(file_path).suffix.lower()
    
    try:
        if file_extension == '.txt':
            try:
                # Read the text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Check if any content was loaded
                if not text.strip():
                    logger.warning(f"No content found in text file: {file_path}")
                    return []
                    
                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "txt",
                        "path": file_path
                    }
                )
                
            except Exception as txt_error:
                logger.error(f"Error loading text file {file_path}: {str(txt_error)}")
                return []
                
        elif file_extension == '.json':
            try:
                # For JSON files, we'll load the entire content and join all values
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Recursively extract all string values from JSON
                def extract_values(obj, arr):
                    if isinstance(obj, dict):
                        for value in obj.values():
                            extract_values(value, arr)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_values(item, arr)
                    elif isinstance(obj, str):
                        arr.append(obj)
                    return arr
                
                text_values = extract_values(json_data, [])
                if not text_values:
                    logger.warning(f"No text content found in JSON file: {file_path}")
                    return []
                
                text_content = "\n".join(text_values)
                
                # Create Document object with metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": "json",
                        "path": file_path
                    }
                )
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Invalid JSON in file {file_path}: {str(json_error)}")
                return []
            except Exception as json_error:
                logger.error(f"Error processing JSON file {file_path}: {str(json_error)}")
                return []
        else:
            logger.warning(f"Unsupported file type: {file_extension} for file: {file_path}")
            return []
            
        # Use standardized text splitter
        text_splitter = get_text_splitter()
        chunks = text_splitter.split_documents([doc])
        
        # Validate and clean the chunks
        valid_chunks = validate_documents(chunks)
        
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

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
            ('div', {'id': 'main'})
        ]
        
        for tag, attrs in content_priority:
            if attrs:
                main_content = soup.find(tag, attrs)
            else:
                main_content = soup.find(tag)
            if main_content:
                break
                
        # Extract text from the main content or the whole page
        if main_content:
            text = main_content.get_text()
            logger.info("Using main content section of the page")
        else:
            text = soup.get_text()
            logger.info("Using full page content (no main section identified)")
            
        # Clean the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Check if any text was extracted
        if not text.strip():
            logger.warning(f"No text content could be extracted from {url}")
            return []
            
        logger.info(f"Extracted {len(text.split())} words from the website")
        
        # Use standardized text splitter
        text_splitter = get_text_splitter()
        
        # Create Document object with metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": url,
                "type": "website",
                "title": soup.title.string if soup.title else "Untitled",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        # Split the document into chunks
        chunks = text_splitter.split_documents([doc])
        
        # Validate and clean the chunks
        valid_chunks = validate_documents(chunks)
        
        logger.info(f"Successfully extracted {len(valid_chunks)} chunks from website")
        return valid_chunks
        
    except Exception as e:
        logger.error(f"Error processing website {url}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def load_and_process_pdfs(data_dir: str, processed_files: Set[str]) -> Tuple[List[Document], Set[str]]:
    """
    Load PDFs from directory and split into chunks, skipping already processed files.
    
    Args:
        data_dir: Directory containing PDF files
        processed_files: Set of already processed file paths
        
    Returns:
        Tuple of (document chunks, successfully processed files)
    """
    # Look for PDFs in the pdf_files subdirectory and any other subdirectories
    pdf_files = glob.glob(os.path.join(data_dir, "pdf_files", "**/*.pdf"), recursive=True)
    pdf_files += glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    
    new_files = set(pdf_files) - processed_files
    
    if not new_files:
        logger.info("No new PDF files to process")
        return [], set()
        
    documents = []
    successfully_processed = set()
    
    for pdf_file in tqdm(new_files, desc="Processing PDF files"):
        try:
            chunks = process_single_pdf(pdf_file)
            if chunks:
                documents.extend(chunks)
                successfully_processed.add(pdf_file)
                logger.info(f"Successfully processed: {pdf_file} ({len(chunks)} chunks)")
            else:
                logger.warning(f"No content could be extracted from {pdf_file}")
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
            
    if not documents:
        logger.warning("No documents were successfully processed")
        
    return documents, successfully_processed

def load_and_process_text_json(data_dir: str, processed_files: Set[str]) -> Tuple[List[Document], Set[str]]:
    """
    Load and process new .txt and .json files from the specified directory.
    
    Args:
        data_dir: Directory containing text and JSON files
        processed_files: Set of already processed file paths
        
    Returns:
        Tuple of (document chunks, successfully processed files)
    """
    # Look in text_files and json_files subdirectories and any other subdirectories
    txt_files = glob.glob(os.path.join(data_dir, "text_files", "**/*.txt"), recursive=True)
    txt_files += glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    
    json_files = glob.glob(os.path.join(data_dir, "json_files", "**/*.json"), recursive=True)
    json_files += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    # Remove duplicates
    txt_files = list(set(txt_files))
    json_files = list(set(json_files))
    
    all_files = set(txt_files + json_files)
    new_files = all_files - processed_files
    
    if not new_files:
        logger.info("No new TXT or JSON files to process")
        return [], set()
        
    chunks = []
    successfully_processed = set()
    
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_file, file_path): file_path
                         for file_path in new_files}
                         
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                          total=len(new_files),
                          desc="Processing TXT/JSON files"):
            file_path = future_to_file[future]
            try:
                file_chunks = future.result()
                if file_chunks:
                    chunks.extend(file_chunks)
                    successfully_processed.add(file_path)
                    logger.info(f"Successfully processed: {file_path} ({len(file_chunks)} chunks)")
                else:
                    logger.warning(f"No content could be extracted from {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
                
    return chunks, successfully_processed

def process_website_input() -> None:
    """
    Interactively process websites for vectorization.
    """
    while True:
        user_input = input("\nWould you like to vectorize a webpage? (Y/N): ").strip().upper()
        
        if user_input == 'N':
            logger.info("Exiting website vectorization.")
            break
        elif user_input == 'Y':
            url = input("Please enter the website URL: ").strip()
            try:
                # Process the website
                logger.info(f"Processing website: {url}")
                chunks = process_website(url)
                
                if not chunks:
                    logger.error("No chunks were created from the website content")
                    print("No content could be extracted from the website")
                    continue
                    
                logger.info(f"Successfully extracted {len(chunks)} chunks from website")
                
                # Add chunks to vector store
                vector_store_manager.add_documents(chunks)
                
                print(f"\nSuccessfully processed website: {url}")
                print(f"Created {len(chunks)} chunks from the website content")
                
            except requests.exceptions.RequestException as req_error:
                logger.error(f"Error fetching website: {str(req_error)}")
                print(f"Error fetching website: {str(req_error)}")
            except Exception as e:
                logger.error(f"Error processing website: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                print(f"Error processing website: {str(e)}")
        else:
            print("Invalid input. Please enter 'Y' or 'N'")

def main() -> None:
    """
    Main function to process documents and add them to the vector store.
    """
    # Create data directory structure if it doesn't exist
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, "pdf_files"), exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, "text_files"), exist_ok=True)
    os.makedirs(os.path.join(config.data_dir, "json_files"), exist_ok=True)
    
    # Get already processed files
    processed_files = get_processed_files(config.db_dir)
    
    # Process PDF files
    pdf_chunks, new_pdf_files = load_and_process_pdfs(config.data_dir, processed_files)
    
    # Process text and JSON files
    text_json_chunks, new_text_files = load_and_process_text_json(config.data_dir, processed_files)
    
    # Combine all chunks
    all_new_chunks = pdf_chunks + text_json_chunks
    
    if all_new_chunks:
        # Add documents to vector store
        vector_store_manager.add_documents(all_new_chunks)
        
        # Update processed files tracking
        update_processed_files(config.db_dir, new_pdf_files | new_text_files)
        
        # Summary statistics
        print("\nIngest Summary:")
        print(f"Total files processed: {len(new_pdf_files) + len(new_text_files)}")
        print(f"- PDF files: {len(new_pdf_files)}")
        print(f"- Text/JSON files: {len(new_text_files)}")
        print(f"Total chunks created: {len(all_new_chunks)}")
        print(f"Average tokens per chunk: ~{len(all_new_chunks[0].page_content.split()) if all_new_chunks else 0} words")
    else:
        print("No new content to process")
        
    # Process websites interactively
    process_website_input()

if __name__ == "__main__":
    main()
