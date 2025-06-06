"""
Error handler for AgenticRAG.
Provides standardized error handling functions and custom exceptions.
"""

import logging
import traceback
import sys
from typing import Optional, Type, Callable, Any, Dict
import streamlit as st

# Set up logging
logger = logging.getLogger(__name__)

# Custom exception classes
class AgenticRAGError(Exception):
    """Base exception class for AgenticRAG."""
    pass

class DocumentProcessingError(AgenticRAGError):
    """Exception raised for errors during document processing."""
    pass

class ModelError(AgenticRAGError):
    """Exception raised for errors related to models."""
    pass

class VectorStoreError(AgenticRAGError):
    """Exception raised for errors related to vector stores."""
    pass

class WebScrapingError(AgenticRAGError):
    """Exception raised for errors during web scraping."""
    pass

class WebSearchError(AgenticRAGError):
    """Exception raised for errors during web search."""
    pass

# Error handling functions
def handle_error(
    error: Exception,
    error_type: Optional[Type[Exception]] = None,
    message: Optional[str] = None,
    log_level: int = logging.ERROR,
    show_traceback: bool = True,
    raise_exception: bool = False,
    streamlit_error: bool = False
) -> Dict[str, Any]:
    """
    Handle an error with standardized logging and optional re-raising.
    
    Args:
        error: The exception that was caught
        error_type: Optional type to convert the error to
        message: Optional message to prepend to the error
        log_level: Logging level to use
        show_traceback: Whether to include the traceback in the log
        raise_exception: Whether to re-raise the exception
        streamlit_error: Whether to display the error in Streamlit
        
    Returns:
        A dictionary with error information
    """
    # Format the error message
    error_message = f"{message + ': ' if message else ''}{str(error)}"
    
    # Get the traceback
    tb = traceback.format_exc() if show_traceback else None
    
    # Log the error
    if log_level == logging.DEBUG:
        logger.debug(error_message)
        if tb:
            logger.debug(f"Traceback: {tb}")
    elif log_level == logging.INFO:
        logger.info(error_message)
        if tb:
            logger.info(f"Traceback: {tb}")
    elif log_level == logging.WARNING:
        logger.warning(error_message)
        if tb:
            logger.warning(f"Traceback: {tb}")
    else:  # Default to ERROR
        logger.error(error_message)
        if tb:
            logger.error(f"Traceback: {tb}")
    
    # Display in Streamlit if requested
    if streamlit_error and 'st' in sys.modules:
        st.error(error_message)
    
    # Create error info dictionary
    error_info = {
        "error": error,
        "message": error_message,
        "traceback": tb,
        "type": type(error).__name__
    }
    
    # Re-raise if requested
    if raise_exception:
        if error_type:
            raise error_type(error_message) from error
        else:
            raise
    
    return error_info

def try_except_decorator(
    error_type: Optional[Type[Exception]] = None,
    message: Optional[str] = None,
    log_level: int = logging.ERROR,
    show_traceback: bool = True,
    raise_exception: bool = False,
    streamlit_error: bool = False,
    default_return: Any = None
):
    """
    Decorator for try-except blocks with standardized error handling.
    
    Args:
        error_type: Optional type to convert the error to
        message: Optional message to prepend to the error
        log_level: Logging level to use
        show_traceback: Whether to include the traceback in the log
        raise_exception: Whether to re-raise the exception
        streamlit_error: Whether to display the error in Streamlit
        default_return: Value to return if an exception occurs and raise_exception is False
        
    Returns:
        A decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Use the function name in the message if none provided
                func_message = message or f"Error in {func.__name__}"
                
                # Handle the error
                handle_error(
                    error=e,
                    error_type=error_type,
                    message=func_message,
                    log_level=log_level,
                    show_traceback=show_traceback,
                    raise_exception=raise_exception,
                    streamlit_error=streamlit_error
                )
                
                # Return the default value if not raising
                return default_return
        return wrapper
    return decorator

# Context manager for error handling
class ErrorHandler:
    """
    Context manager for standardized error handling.
    
    Example:
        with ErrorHandler("Processing document", streamlit_error=True):
            process_document(doc)
    """
    
    def __init__(
        self,
        message: Optional[str] = None,
        error_type: Optional[Type[Exception]] = None,
        log_level: int = logging.ERROR,
        show_traceback: bool = True,
        raise_exception: bool = False,
        streamlit_error: bool = False
    ):
        """
        Initialize the error handler.
        
        Args:
            message: Optional message to prepend to the error
            error_type: Optional type to convert the error to
            log_level: Logging level to use
            show_traceback: Whether to include the traceback in the log
            raise_exception: Whether to re-raise the exception
            streamlit_error: Whether to display the error in Streamlit
        """
        self.message = message
        self.error_type = error_type
        self.log_level = log_level
        self.show_traceback = show_traceback
        self.raise_exception = raise_exception
        self.streamlit_error = streamlit_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            # Handle the error
            handle_error(
                error=exc_val,
                error_type=self.error_type,
                message=self.message,
                log_level=self.log_level,
                show_traceback=self.show_traceback,
                raise_exception=self.raise_exception,
                streamlit_error=self.streamlit_error
            )
            
            # Return True to suppress the exception if not raising
            return not self.raise_exception
        
        return False
