#!/bin/bash
set -e

# ASCII art banner
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             DOWNLOAD MISTRAL MODEL FOR URA                 ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check if the models directory exists
if [ ! -d "models" ]; then
  echo "📁 Creating models directory..."
  mkdir -p models
fi

MODEL_PATH="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
  echo "✅ Model already exists at $MODEL_PATH"
  echo "   If you want to redownload, please delete it first."
  exit 0
fi

# Download the model
echo "🔄 Downloading Mistral 7B Instruct model (Q4_K_M quantization)..."
echo "   This will take some time depending on your internet connection."
echo "   File size is approximately 4.7GB."

# Use curl with progress bar
curl -L -o "$MODEL_PATH" \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Check if download was successful
if [ -f "$MODEL_PATH" ]; then
  echo "✅ Model downloaded successfully to $MODEL_PATH"
  echo "   You can now run the URA pipeline with this model."
else
  echo "❌ Failed to download the model. Please check your internet connection and try again."
  exit 1
fi 