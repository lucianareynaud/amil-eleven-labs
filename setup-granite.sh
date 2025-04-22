#!/bin/bash
set -e

echo "===== Setting up Granite3.1-MoE 1B model ====="

# Create Granite modelfile
echo "Creating Granite modelfile..."
cat > /tmp/granite-modelfile.txt << 'EOL'
FROM granite3.1-moe:1b

SYSTEM """
You are a helpful voice assistant called PureVoice. 
You have access to a vast knowledge base and can answer questions on many topics.
Never say you don't have certain capabilities or access to information.
Always try to provide a helpful, concise response.
If you truly don't know something, you can say "I'm not sure about that" instead of saying you lack capabilities.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"
EOL

# Pull the base model
echo "Downloading granite3.1-moe:1b base model..."
ollama pull granite3.1-moe:3b

# Create the custom model
echo "Creating purevoice-granite model..."
ollama create purevoice-granite -f /tmp/granite-modelfile.txt

# Warm up the model
echo "Warming up the model..."
ollama run purevoice-granite "Write a short greeting" --verbose=false

echo "===== Granite3.1-MoE 3B model setup complete =====" 