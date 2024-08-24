#!/bin/bash

echo "Al Rashid Issa as a Machine Learning Engineer, He's just starting out!"
echo "Activating virtual environment..."
cd ~/Desktop/BreastCancer || { echo "Failed to change directory to ~/Desktop/BreastCancer"; exit 1; }
source env_/bin/activate || { echo "Failed to activate the virtual environment"; exit 1; }

# Initialize variables
YAML_PATH=""
ANALAYZR=""
TRAIN="False"
MLFLOW="False"

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --yaml_path)
            YAML_PATH="$2"
            shift
            ;;
        --analyzer)
            ANALAYZR="True"  # Set to True if --analyzer is present
            ;;
        --train)
            TRAIN="$2"
            shift
            ;;
        --mlflow)
            MLFLOW="$2"
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

# Function to start the Django server
start_django_server() {
    echo "Starting Django API server..."
    cd ~/Desktop/BreastCancer/API_web/BreastCancerAPI || { echo "Failed to change directory to ~/Desktop/BreastCancer/API_web/BreastCancerAPI"; exit 1; }
    echo "Running Django server on port 8000..."
    python manage.py runserver 8000 &
}

# Position 1: Train only
if [ "$TRAIN" = "True" ] && [ "$MLFLOW" = "False" ]; then
    echo "Running training process..."
    cd ~/Desktop/BreastCancer || { echo "Failed to change directory to ~/Desktop/BreastCancer"; exit 1; }
    if [ "$ANALAYZR" = "True" ]; then
        echo "Running main.py with --yaml_path and --analyzer arguments"
        python main.py --yaml_path "$YAML_PATH" --analyzer
    else
        echo "Running main.py with --yaml_path argument"
        python main.py --yaml_path "$YAML_PATH"
    fi
    start_django_server
    exit 0
fi


# Position 3: Train and MLflow
if [ "$TRAIN" = "True" ] && [ "$MLFLOW" = "True" ]; then
    echo "MLflow configuration is set, but the MLflow server is not being started by this script."
    echo "Running mlflow_pipeline.py..."
    cd ~/Desktop/BreastCancer/MLflow || { echo "Failed to change directory to ~/Desktop/BreastCancer/MLflow"; exit 1; }
    python mlflow_pipeline.py --yaml_path "$YAML_PATH"

    start_django_server
    exit 0
fi

# Position 4: API only
if [ "$TRAIN" = "False" ] && [ "$MLFLOW" = "False" ]; then
    start_django_server
    exit 0
fi

echo "No valid options provided. Exiting."
exit 1
