import subprocess
import os
import sys

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return f"Error: {result.stderr.strip()}"

def change_directory(directory):
    try:
        directory = os.path.expanduser(directory)
        os.chdir(directory)
        return f"Changed directory to {directory}"
    except FileNotFoundError:
        return f"Error: Failed to change directory to {directory}"

def activate_virtual_environment(env_path):
    activate_script = os.path.join(os.path.expanduser(env_path), 'bin', 'activate')
    if os.path.exists(activate_script):
        command = f"source {activate_script} && echo 'Virtual environment activated'"
        return run_command(command)
    else:
        return "Error: Failed to activate the virtual environment"

def run_training_pipeline(config, analyzer):
    messages = []
    messages.append("Starting training process...")
    messages.append(change_directory('~/Desktop/BreastCancer/src/training'))
    command = ["python", "train_pipeline.py", "--config", config]
    if analyzer:
        command.append("--analyzer")
    messages.append(run_command(" ".join(command)))
    return messages

def run_mlflow_pipeline(config):
    messages = []
    messages.append("MLflow configuration is enabled. Starting MLflow pipeline...")
    messages.append(change_directory('~/Desktop/BreastCancer/src/mlflow'))
    messages.append(run_command(f"python mlflow_pipeline.py --config {config}"))
    return messages

def run_mlflow_ui(port=8080):
    command = f"mlflow ui --port {port}"
    return run_command(command)

def main(config, analyzer=False, train=False, mlflow=False, mlflow_ui=False):
    messages = []
    
    # Change to project directory
    messages.append("Changing directory to the BreastCancer project directory...")
    messages.append(change_directory('~/Desktop/BreastCancer'))

    # Activate virtual environment
    messages.append(activate_virtual_environment('~/Desktop/BreastCancer/env_'))

    # Validate CONFIG if --train is True
    if train and not config:
        messages.append("Error: --config is required when --train is True.")
        return messages

    # Run MLflow UI if requested
    if mlflow_ui:
        messages.append("Starting MLflow UI...")
        messages.append(run_mlflow_ui())

    # Position 1: Train only
    if train and not mlflow:
        messages.extend(run_training_pipeline(config, analyzer))
    
    # Position 2: MLflow pipeline
    elif train and mlflow:
        messages.extend(run_mlflow_pipeline(config))

    return messages

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the BreastCancer project operations")
    parser.add_argument("--config", type=str, required=False, help="Path to the configuration file")
    parser.add_argument("--analyzer", action="store_true", help="Enable the analyzer")
    parser.add_argument("--train", action="store_true", help="Enable training")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow")
    parser.add_argument("--mlflow-ui", action="store_true", help="Run the MLflow UI")

    args = parser.parse_args()

    output_messages = main(config=args.config, analyzer=args.analyzer, train=args.train, mlflow=args.mlflow, mlflow_ui=args.mlflow_ui)

    # Print or log the collected messages
    for message in output_messages:
        print(message)