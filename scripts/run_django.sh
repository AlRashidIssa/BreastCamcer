#!/bin/bash

# Name of Django project
PROJECT_NAME="BreastCancerAPI"

# Default values
HOST="127.0.0.1"
PORT="8000"
VENV_DIR="env_"
MANAGE_PY="manage.py"

# Change directory to the project base directory
cd ~/Desktop/BreastCancer || { echo "Failed to change directory to ~/Desktop/BreastCancer"; exit 1; }

# Activate virtual environment
if [ "$VENV_DIR" ]; then
    echo "Activating virtual environment in $(pwd)..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment directory $VENV_DIR not found in $(pwd). Please ensure it is set up."
    exit 1
fi

# Change directory to Django project
echo "Changing directory to Django project..."
cd ~/Desktop/BreastCancer/src/api/BreastCancerAPI || { echo "Failed to change directory to Django project"; exit 1; }
echo "Current directory: $(pwd)"

# Run Django development server
echo "Running Django development server..."
python "$MANAGE_PY" runserver "$HOST:$PORT"

# Deactivate virtual environment after the server stops
deactivate
echo "Virtual environment deactivated."
