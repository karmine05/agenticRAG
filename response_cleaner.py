"""
Response cleaner module for AgenticRAG.
Provides functions to clean responses from the model.
"""

import re

def clean_response(response):
    """
    Clean the response by removing Call ID fields, error messages, and other debugging information.
    
    Args:
        response: The response string to clean
        
    Returns:
        A cleaned response string or a fallback message if the response contains critical errors
    """
    if not isinstance(response, str):
        # If it's not a string, try to convert it
        try:
            response = str(response)
        except:
            return "I apologize, but I encountered an error processing your request. Please try again."
    
    # Check for critical Ollama server errors
    if "Error code: 500" in response or "HTTP/1.1 500 Internal Server Error" in response:
        return "I apologize, but there seems to be an issue with the AI model server. Please try again in a moment or contact support if the issue persists."
    
    # Check for EOF errors which indicate Ollama server issues
    if "EOF" in response and "error" in response:
        return "I apologize, but there seems to be a connection issue with the AI model server. Please try again in a moment."
        
    # Check for string indices errors
    if "string indices must be integers" in response:
        return "I apologize, but there was an error processing your request. Please try again."
        
    # Check for New run markers from smolagents
    if "New run" in response and "OpenAIServerModel" in response:
        return "I apologize, but there was an error with the model. Please try again with a different question or contact support."
    
    # Remove Step markers with timestamps
    response = re.sub(r'━+\s*Step \d+\s*━+', '', response)
    
    # Remove Call ID fields (e.g., "Call id: call_11")
    response = re.sub(r'\[?Call id: call_\d+\]?', '', response)
    
    # Remove error messages
    response = re.sub(r'Error:.*?\n', '', response)
    response = re.sub(r'ERROR:.*?\n', '', response)
    response = re.sub(r'WARNING:.*?\n', '', response)
    response = re.sub(r'\[Step \d+: Duration.*?\]', '', response)
    response = re.sub(r'Error in generating model output:.*?\n', '', response)
    
    # Remove HTTP request logs
    response = re.sub(r'INFO:httpx:HTTP Request:.*?\n', '', response)
    
    # Remove openai client logs
    response = re.sub(r'INFO:openai.*?\n', '', response)
    
    # Remove function call debugging info
    response = re.sub(r'\[\{.*?\'function\'.*?\}\]', '', response)
    
    # Remove any lines with "Reached max steps"
    response = re.sub(r'Reached max steps\..*?\n', '', response)
    
    # Remove any lines with "INFO:" 
    response = re.sub(r'INFO:.*?\n', '', response)
    
    # Remove model manager logs
    response = re.sub(r'WARNING:model_manager:.*?\n', '', response)
    response = re.sub(r'ERROR:model_manager:.*?\n', '', response)
    
    # Remove improved_agent_rag logs
    response = re.sub(r'INFO:improved_agent_rag:.*?\n', '', response)
    response = re.sub(r'WARNING:improved_agent_rag:.*?\n', '', response)
    response = re.sub(r'ERROR:improved_agent_rag:.*?\n', '', response)
    
    # Remove New run sections
    response = re.sub(r'╭+.*?New run.*?╮[\s\S]*?╰+.*?╯', '', response)
    
    # Clean up multiple newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Trim whitespace
    response = response.strip()
    
    # If after cleaning, the response is empty or very short, provide a fallback
    if not response or len(response) < 10:
        return "I apologize, but I couldn't generate a proper response. Please try again with a different question."
    
    return response
