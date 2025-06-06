#!/bin/bash

# Script to restart the Ollama server
# This can be run when the server encounters issues

echo "Attempting to restart Ollama server..."

# Check if Ollama is running
if pgrep -x "ollama" > /dev/null
then
    echo "Stopping Ollama service..."
    pkill -f ollama
    sleep 2
else
    echo "Ollama service is not running."
fi

# Start Ollama
echo "Starting Ollama service..."
ollama serve > /dev/null 2>&1 &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 5

# Check if Ollama is now running
if pgrep -x "ollama" > /dev/null
then
    echo "Ollama service has been restarted successfully."
    echo "You can now return to your application."
else
    echo "Failed to restart Ollama service."
    echo "Please try starting it manually with 'ollama serve'."
fi
