#!/bin/bash

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install reportlab
echo "Installing reportlab..."
pip install reportlab==4.0.9

echo "Installation complete. Please try running your application again."