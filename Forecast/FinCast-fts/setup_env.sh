#!/bin/bash


# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing it now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create a virtual environment
echo "Creating virtual environment..."
uv venv .venv

# Activate the virtual environment
source .venv/Scripts/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt


# Download model weights
MODEL_PATH="Inference/v1.pth"
MODEL_URL="https://huggingface.co/Vincent05R/FinCast/resolve/main/v1.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model weights to $MODEL_PATH..."
    # Create directory if it doesn't exist
    mkdir -p Inference
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
else
    echo "Model weights already exist at $MODEL_PATH"
fi

echo "Setup complete! To activate the environment, run: source .venv/Scripts/activate"
