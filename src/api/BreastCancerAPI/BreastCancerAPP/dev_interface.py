import subprocess
import os

def run_command(command, output_file=None):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")
    output = result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    
    if output_file:
        with open(output_file, 'a') as f:
            f.write(output + '\n')
    return output

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
    messages.append(change_directory('~/app/src/training'))
    command = ["python", "train_pipeline.py", "--config", config]
    if analyzer:
        command.append("--analyzer")
    messages.append(run_command(" ".join(command)))
    return messages

def run_mlflow_pipeline(config):
    messages = []
    messages.append("MLflow configuration is enabled. Starting MLflow pipeline...")
    messages.append(change_directory('~/app/src/mlflow'))
    messages.append(run_command(f"python mlflow_pipeline.py --config {config}"))
    return messages

def run_mlflow_ui(port=8080, output_file="mlflow_ui_output.txt"):
    command = f"mlflow ui --port {port}"
    return run_command(command, output_file=output_file)

def run_main_function(config, analyzer=False, train=False, mlflow=False, mlflow_ui=False):
    messages = []
    
    # Change to project directory
    messages.append("Changing directory to the BreastCancer project directory...")
    messages.append(change_directory('~/app'))

    # Activate virtual environment
    messages.append(activate_virtual_environment('~/app/env_'))

    # Validate CONFIG if --train is True
    if train and not config:
        messages.append("Error: --config is required when --train is True.")
        return messages

    # Run MLflow UI if requested
    if mlflow_ui:
        messages.append("Starting MLflow UI...")
        output_file = "mlflow_ui_output.txt"
        messages.append(run_command(f"mlflow ui --port 8080", output_file=output_file))

    # Position 1: Train only
    if train and not mlflow:
        messages.extend(run_training_pipeline(config, analyzer))
    
    # Position 2: MLflow pipeline
    elif train and mlflow:
        messages.extend(run_mlflow_pipeline(config))

    return messages
