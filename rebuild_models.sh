#!/bin/bash

# Set environment variables for Ollama optimization
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_KV_CACHE_TYPE=f16

echo "Rebuilding Ollama models with optimized settings..."

# Rebuild the Deepseek-r1-14b-64k-ab model (REASONING_MODEL_ID)
echo "Rebuilding Deepseek-r1-14b-64k-ab model..."
ollama create deepseek-r1:14b-64k-ab -f ollama_models/Deepseek-r1-14b-64k-ab

# Rebuild the Qwen-7b-Instruct-64K model (TOOL_MODEL_ID and IOC_TOOL_MODEL_ID)
echo "Rebuilding Qwen-7b-Instruct-64K model..."
ollama create qwen2.5:7b-64k -f ollama_models/Qwen-7b-Instruct-64K

# Rebuild the Llama-3-8b-64k model (GENERAL_TOOL_MODEL_ID)
echo "Rebuilding Llama-3-8b-64k model..."
ollama create llama3:8b-64k -f ollama_models/Llama-3-8b-64k

echo "All required models have been rebuilt with optimized settings."
