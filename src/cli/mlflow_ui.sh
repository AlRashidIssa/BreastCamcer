# Display message indicating the start of directory change
echo "Changing directory to the BreastCancer project directory..."

# Change to the project directory
cd ~/Desktop/BreastCancer

# Activate the virtual environment
echo "Activating the virtual environment..."
source env_/bin/activate

# Start MLflow UI on port 8080
echo "Starting MLflow UI on port 8080..."
mlflow ui --port 8080
