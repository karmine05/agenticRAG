"""
Utility to convert web search results to a temporary vector database.
"""

import logging
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from document_processor import create_temp_vector_store

# Set up logging
logger = logging.getLogger(__name__)

def web_results_to_temp_db(web_results: List[Dict[str, Any]]) -> Optional[Tuple[Any, str]]:
    """
    Convert web search results to a temporary vector database.

    Args:
        web_results: List of web search results

    Returns:
        Tuple of (temporary vector database, temporary directory path) or None if failed
    """
    if not web_results:
        logger.warning("No web search results to convert to temporary database")
        return None

    try:
        # Create documents from web results with improved formatting
        web_docs = []
        for result in web_results:
            title = result.get('title', '')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            # Enhanced content formatting to improve retrieval quality
            # Include more structured information and clear section markers
            content = f"""
TITLE: {title}

SOURCE: {link}

SUMMARY: {snippet}

CONTENT TYPE: Web Search Result
            """

            # Create a document with enhanced metadata
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "source": link,
                    "type": "web_search",
                    "title": title,
                    "timestamp": "current",
                    "source_db": "temp",  # Explicitly mark as temporary database source
                    "priority": "high"    # Mark web search results as high priority
                }
            )
            web_docs.append(doc)

        logger.info(f"Created {len(web_docs)} enhanced documents from web search results")

        # Create a temporary vector store with these documents
        temp_vectordb, temp_db_dir = create_temp_vector_store(web_docs)
        logger.info(f"Successfully created temporary vector store at {temp_db_dir}")

        # Add to session state if available
        if 'st' in globals() and hasattr(st, 'session_state'):
            if not hasattr(st.session_state, 'temp_vectordbs'):
                st.session_state.temp_vectordbs = []

            # Add to the beginning of the list to prioritize most recent web search results
            st.session_state.temp_vectordbs.insert(0, temp_vectordb)
            logger.info(f"Added temporary vector store to session state (now have {len(st.session_state.temp_vectordbs)} temp DBs)")

            # Limit the number of temporary databases to prevent memory issues
            if len(st.session_state.temp_vectordbs) > 5:
                # Remove the oldest temporary database
                old_db = st.session_state.temp_vectordbs.pop()
                logger.info(f"Removed oldest temporary database to maintain limit of 5")

        return temp_vectordb, temp_db_dir

    except Exception as e:
        logger.error(f"Error creating temporary vector store from web results: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
