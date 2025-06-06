"""
Model manager for AgenticRAG.
Provides a unified interface for model initialization, calling, and error handling.
"""

import os
import logging
import time
import subprocess
import json
import traceback
from typing import Any, Dict, Optional, Union, List

# Set up logging
logger = logging.getLogger(__name__)

def robust_model_call(model: Any, prompt: str, max_retries: int = 3, retry_delay: int = 2) -> str:
    """
    Call a model with robust error handling and retries.
    
    Args:
        model: The model to call
        prompt: The prompt to send to the model
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        The model's response as a string
    """
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            # Try calling the model directly
            logger.debug(f"Calling model with prompt: {prompt[:100]}...")
            
            # Print model type for debugging
            logger.debug(f"Model type: {type(model).__name__}")
            
            # Try to call the model safely
            try:
                # Format the prompt as a message if needed
                if hasattr(model, 'client') and hasattr(model, '_prepare_completion_kwargs'):
                    # This is likely an OpenAIServerModel from smolagents
                    # Format the prompt as a message
                    messages = [{"role": "user", "content": prompt}]
                    response = model.client.chat.completions.create(
                        model=model.model_id,
                        messages=messages
                    )
                    logger.debug(f"Response type: {type(response)}")
                    
                    # Extract the content from the response
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                            return response.choices[0].message.content
                    
                    # Fallback to string representation
                    return str(response)
                else:
                    # For other model types, call directly
                    response = model(prompt)
                    logger.debug(f"Response type: {type(response)}")
                    
                    # If response is a string, return it directly
                    if isinstance(response, str):
                        return response
                    
                    # Try to extract content from various response formats
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                            return response.choices[0].message.content
                        elif hasattr(response.choices[0], 'text'):
                            return response.choices[0].text
                    
                    # Try other common response attributes
                    if hasattr(response, 'content'):
                        return response.content
                    elif hasattr(response, 'text'):
                        return response.text
                    elif hasattr(response, 'output'):
                        return response.output
                    
                    # Try to convert to string as a last resort
                    return str(response)
                
            except Exception as call_error:
                logger.debug(f"Error calling model: {str(call_error)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
                
        except Exception as e:
            last_error = e
            retries += 1
            
            if retries <= max_retries:
                logger.warning(f"Error calling model (attempt {retries}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to call model after {max_retries} attempts: {str(e)}")
                raise
    
    # This should not be reached, but just in case
    raise last_error or ValueError("Unknown error in robust_model_call")

def get_model_with_fallback(model_id: str, api_token: Optional[str] = None) -> Any:
    """
    Get a model with fallback options.
    
    Args:
        model_id: The ID of the model to get
        api_token: Optional API token for the model service
        
    Returns:
        The model instance
    """
    # Check if we should use HuggingFace
    use_huggingface = os.getenv("USE_HUGGINGFACE", "no").lower() in ["yes", "true", "1"]
    
    if use_huggingface:
        logger.info(f"Using HuggingFace for model: {model_id}")
        
        try:
            # Try to import HfApiModel
            try:
                from smolagents import HfApiModel
            except ImportError:
                logger.error("Failed to import HfApiModel from smolagents")
                raise
            
            # Initialize the model
            model = HfApiModel(
                model_id=model_id,
                api_token=api_token or os.getenv("HUGGINGFACE_API_TOKEN", "")
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFace model {model_id}: {str(e)}")
            raise
    else:
        logger.info(f"Using Ollama for model: {model_id}")
        
        try:
            # Try to import OpenAIServerModel
            try:
                from smolagents import OpenAIServerModel
                from openai import OpenAI
            except ImportError:
                logger.error("Failed to import OpenAIServerModel from smolagents or OpenAI")
                raise
            
            # Check if Ollama is running
            try:
                subprocess.run(["curl", "-s", "http://localhost:11434/api/version"], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                logger.warning("Ollama server not running. Attempting to start it...")
                try:
                    # Try to start Ollama
                    subprocess.Popen(["ollama", "serve"], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                    # Wait for Ollama to start
                    time.sleep(5)
                except Exception as start_error:
                    logger.error(f"Failed to start Ollama: {str(start_error)}")
            
            # Create a direct OpenAI client for Ollama
            client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't need a real API key
            )
            
            # Create a simple wrapper class
            class OllamaWrapper:
                def __init__(self, client, model_id):
                    self.client = client
                    self.model_id = model_id
                
                def __call__(self, prompt):
                    # Format the prompt as a message
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages
                    )
                    
                    # Extract the content from the response
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                            return response.choices[0].message.content
                    
                    # Fallback to string representation
                    return str(response)
            
            # Create the wrapper
            model = OllamaWrapper(client, model_id)
            
            return model
            
        except Exception as e:
            logger.error(f"Error initializing Ollama model {model_id}: {str(e)}")
            raise

class ModelManager:
    """
    A unified interface for managing models in AgenticRAG.
    Handles model initialization, calling, and error handling.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.model_configs = {}
        
    def initialize_model(self, model_id: str, model_type: str, api_token: Optional[str] = None) -> str:
        """
        Initialize a model and store it for later use.
        
        Args:
            model_id: The ID of the model to initialize
            model_type: The type of model ('reasoning' or 'tool')
            api_token: Optional API token for the model service
            
        Returns:
            A unique key for accessing the model
        """
        logger.info(f"Initializing {model_type} model: {model_id}")
        
        try:
            # Get the model with fallback options
            model = get_model_with_fallback(model_id, api_token=api_token)
            
            # Generate a unique key for this model
            key = f"{model_type}_{model_id}_{id(model)}"
            
            # Store the model and its configuration
            self.models[key] = model
            self.model_configs[key] = {
                'model_id': model_id,
                'model_type': model_type,
                'api_token': api_token,
                'initialized_at': time.time()
            }
            
            logger.info(f"Successfully initialized {model_type} model: {model_id}")
            return key
            
        except Exception as e:
            logger.error(f"Error initializing {model_type} model {model_id}: {str(e)}")
            raise
    
    def get_model(self, key: str) -> Any:
        """
        Get a model by its key.
        
        Args:
            key: The unique key for the model
            
        Returns:
            The model instance
        """
        if key not in self.models:
            raise ValueError(f"Model with key {key} not found")
        
        return self.models[key]
    
    def call_model(self, key: str, prompt: str, max_retries: int = 3, retry_delay: int = 2) -> str:
        """
        Call a model with robust error handling.
        
        Args:
            key: The unique key for the model
            prompt: The prompt to send to the model
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            The model's response as a string
        """
        if key not in self.models:
            raise ValueError(f"Model with key {key} not found")
        
        model = self.models[key]
        
        try:
            # Use the robust_model_call function
            return robust_model_call(model, prompt, max_retries=max_retries, retry_delay=retry_delay)
        except Exception as e:
            logger.error(f"Error calling model {key}: {str(e)}")
            
            # Try to reinitialize the model if the call fails
            try:
                config = self.model_configs[key]
                logger.info(f"Attempting to reinitialize model {config['model_id']}")
                
                # Get a fresh model instance
                new_model = get_model_with_fallback(config['model_id'], api_token=config['api_token'])
                
                # Replace the old model
                self.models[key] = new_model
                
                # Try again with the new model
                return robust_model_call(new_model, prompt, max_retries=max_retries, retry_delay=retry_delay)
            except Exception as reinit_error:
                logger.error(f"Error reinitializing model {key}: {str(reinit_error)}")
                raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all initialized models.
        
        Returns:
            A list of model configurations
        """
        return [
            {
                'key': key,
                **config
            }
            for key, config in self.model_configs.items()
        ]
    
    def remove_model(self, key: str) -> bool:
        """
        Remove a model from the manager.
        
        Args:
            key: The unique key for the model
            
        Returns:
            True if the model was removed, False otherwise
        """
        if key not in self.models:
            return False
        
        try:
            # Remove the model and its configuration
            del self.models[key]
            del self.model_configs[key]
            return True
        except Exception as e:
            logger.error(f"Error removing model {key}: {str(e)}")
            return False

# Create a singleton instance
model_manager = ModelManager()

def initialize_models(reasoning_model_id: str, tool_model_id: str, api_token: Optional[str] = None) -> Dict[str, str]:
    """
    Initialize the reasoning and tool models.
    
    Args:
        reasoning_model_id: The ID of the reasoning model
        tool_model_id: The ID of the tool model
        api_token: Optional API token for the model service
        
    Returns:
        A dictionary with keys for the reasoning and tool models
    """
    reasoning_key = model_manager.initialize_model(reasoning_model_id, 'reasoning', api_token)
    tool_key = model_manager.initialize_model(tool_model_id, 'tool', api_token)
    
    return {
        'reasoning': reasoning_key,
        'tool': tool_key
    }

def call_reasoning_model(model_key: str, prompt: str) -> str:
    """
    Call the reasoning model with robust error handling.
    
    Args:
        model_key: The key for the reasoning model
        prompt: The prompt to send to the model
        
    Returns:
        The model's response as a string
    """
    return model_manager.call_model(model_key, prompt, max_retries=3, retry_delay=2)

def call_tool_model(model_key: str, prompt: str) -> str:
    """
    Call the tool model with robust error handling.
    
    Args:
        model_key: The key for the tool model
        prompt: The prompt to send to the model
        
    Returns:
        The model's response as a string
    """
    return model_manager.call_model(model_key, prompt, max_retries=3, retry_delay=3)
