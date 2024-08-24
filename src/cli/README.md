
### 1. **Setup Your Environment**

Make sure you have the following:
- The `BreastCancer` directory with the correct subdirectories and files.
- A virtual environment named `env_` in `~/Desktop/BreastCancer/`.
- Relevant Python scripts (`train_pipeline.py`, `mlflow_pipeline.py`, etc.) in their expected locations.

### 2. **Script Preparation**

Save the script to a file, e.g., `manage_project.sh`:

```bash
#!/bin/bash

echo "Al Rashid Issa as a Machine Learning Engineer, He's just starting out!"
echo "Activating virtual environment..."

# Change directory to the project folder
cd ~/Desktop/BreastCancer || { echo "Error: Failed to change directory to ~/Desktop/BreastCancer"; exit 1; }

# Activate the virtual environment
source env_/bin/activate || { echo "Error: Failed to activate the virtual environment"; exit 1; }

# Initialize variables
CONFIG="/home/alrashidissa/Desktop/BreastCancer/configs/config.yaml"
ANALAYZR="False"
TRAIN="False"
MLFLOW="False"

# Parse the arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --config)
            CONFIG="$2"
            echo "Configuration file set to: $CONFIG"
            shift
            ;;
        --analyzer)
            ANALAYZR="True"
            echo "Analyzer enabled."
            ;;
        --train)
            TRAIN="$2"
            echo "Training set to: $TRAIN"
            shift
            ;;
        --mlflow)
            MLFLOW="$2"
            echo "MLflow set to: $MLFLOW"
            shift
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Validate CONFIG if --train is True
if [ "$TRAIN" = "True" ] && [ -z "$CONFIG" ]; then
    echo "Error: --config is required when --train is True."
    exit 1
fi

# Function to start the Django server
start_django_server() {
    echo "Starting Django API server..."
    cd ~/Desktop/BreastCancer/src/api/BreastCancerAPI || { echo "Error: Failed to change directory to ~/Desktop/BreastCancer/src/api/BreastCancerAPI"; exit 1; }
    echo "Running Django server on port 8000..."
    python manage.py runserver 8000 &
}

# Position 1: Train only
if [ "$TRAIN" = "True" ] && [ "$MLFLOW" = "False" ]; then
    echo "Starting training process..."
    cd ~/Desktop/BreastCancer/src/training || { echo "Error: Failed to change directory to ~/Desktop/BreastCancer/src/training"; exit 1; }
    if [ "$ANALAYZR" = "True" ]; then
        echo "Running train_pipeline.py with --config and --analyzer arguments"
        python train_pipeline.py --config "$CONFIG" --analyzer
    else
        echo "Running train_pipeline.py with --config argument"
        python train_pipeline.py --config "$CONFIG"
    fi
    start_django_server
    exit 0
fi

# Position 3: Train and MLflow
if [ "$TRAIN" = "True" ] && [ "$MLFLOW" = "True" ]; then
    echo "MLflow configuration is enabled. Starting MLflow pipeline..."
    echo "Running mlflow_pipeline.py with --config argument..."
    cd ~/Desktop/BreastCancer/src/mlflow || { echo "Error: Failed to change directory to ~/Desktop/BreastCancer/src/mlflow"; exit 1; }
    python mlflow_pipeline.py --config "$CONFIG"
    start_django_server
    exit 0
fi

# Position 4: API only
if [ "$TRAIN" = "False" ] && [ "$MLFLOW" = "False" ]; then
    echo "No training or MLflow specified. Starting Django API server..."
    start_django_server
    exit 0
fi

# No valid options provided
echo "Error: No valid options provided. Exiting."
exit 1
```

### 3. **Make the Script Executable**

Run the following command to make your script executable:

```bash
chmod +x manage_project.sh
```

### 4. **Run the Script in Different Scenarios**

#### **Scenario 1: Training Only**

```bash
./manage_project.sh --train True --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml
```

#### **Scenario 2: Training with Analyzer**

```bash
./manage_project.sh --train True --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml --analyzer
```

#### **Scenario 3: Training with MLflow**

```bash
./manage_project.sh --train True --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml --mlflow True
```

#### **Scenario 4: API Only**

```bash
./manage_project.sh
```

#### **Scenario 5: Invalid Parameters**

```bash
./manage_project.sh --unknown_param
```

### **Verify Output**

- **Successful Execution**: Check the terminal output for messages indicating the success of each step.
- **Errors**: Ensure errors are correctly reported for directory changes, environment activation, and unknown parameters.