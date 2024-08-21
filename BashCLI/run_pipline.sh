#!/bin/bash

# Check if the YAML path argument is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 --yaml_path path/to/your/config.yaml [--analayzr analyzer_option]"
    exit 1
fi

# Initialize variables
YAML_PATH=""
ANALAYZR=""

# Parse the YAML path argument and optional analyzer argument
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
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Validate YAML_PATH is set
if [ -z "$YAML_PATH" ]; then
    echo "Error: --yaml_path is required."
    exit 1
fi

# Define the path to your Python script
PYTHON_SCRIPT="/home/alrashidissa/Desktop/BreastCancer/main.py"

# Execute the Python script with the YAML path and optional analyzer
if [ -n "$ANALAYZR" ]; then
    python "$PYTHON_SCRIPT" --yaml_path "$YAML_PATH" --analayzr "$ANALAYZR"
else
    python "$PYTHON_SCRIPT" --yaml_path "$YAML_PATH"
fi
