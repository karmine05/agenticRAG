#!/bin/bash

# Set environment variables for Ollama optimization
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_KV_CACHE_TYPE=f16

# Set environment variables for Python optimization
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "Starting CTI AgenticRAG with optimized settings..."

# Check if OpenAI package is installed
if ! pip3 show openai > /dev/null; then
    echo "Installing missing dependencies..."

    # Activate the conda environment
    conda activate cti-agenticRAG

    # Install the OpenAI package
    pip3 install openai

    # Install smolagents with OpenAI support
    pip3 install 'smolagents[openai]'

    echo "Dependencies installed successfully."
fi

# Start the application
python -m streamlit run web_ui.py
