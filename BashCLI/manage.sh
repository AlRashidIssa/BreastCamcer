#!/bin/bash

echo "Activating virtual environment..."
cd ~/Desktop/BreastCancer
source env_/bin/activate

# Initialize variables
YAML_PATH=""
ANALAYZR=""
TRAIN="False"

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --yaml_path)
            YAML_PATH="$2"
            shift
            ;;
        --analayzr)
            ANALAYZR="$2"
            shift
            ;;
        --train)
            TRAIN="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Validate YAML_PATH if --train is True
if [ "$TRAIN" = "True" ] && [ -z "$YAML_PATH" ]; then
    echo "Error: --yaml_path is required when --train is True."
    exit 1
fi

# Define the main Python script
PYTHON_SCRIPT="/home/alrashidissa/Desktop/BreastCancer/main.py"

# Run training or just API based on the --train parameter
if [ "$TRAIN" = "True" ]; then
    echo "Running training process..."
    if [ -n "$ANALAYZR" ]; then
        python "$PYTHON_SCRIPT" --yaml_path "$YAML_PATH" --analayzr "$ANALAYZR" 
    else
        python "$PYTHON_SCRIPT" --yaml_path "$YAML_PATH"
    fi
else
    echo "Starting API server..."
    # Run the Django API
    echo "Changing directory to the Django project..."
    cd ~/Desktop/BreastCancer/API_web/BreastCancerAPI || exit

    echo "Running Django server..."
    python manage.py runserver
fi
