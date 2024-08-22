Certainly! Here's a revised version of your `README.md` file with a more restrictive license that aligns with your requirements:

---

# Breast Cancer Prediction Pipeline

This repository provides a comprehensive pipeline for processing breast cancer data, training a machine learning model, and evaluating predictions. The pipeline includes components for data ingestion, preprocessing, feature selection, model training, and evaluation. It also integrates with MLflow for experiment tracking and model management.

## Table of Contents

- [License](#license)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). 

**Terms of License:**

- **NonCommercial**: You may not use this work for commercial purposes. Commercial use includes any activity for commercial advantage or monetary compensation.
- **Attribution**: You must give appropriate credit to the original creator, provide a link to the license, and indicate if changes were made.

**Full License Text:**

You can view the full terms of the license [here](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Prerequisites

Before running the pipeline, ensure you have the following installed:

- Python 3.x
- Virtualenv (for managing Python environments)
- [MLflow](https://mlflow.org/) (for experiment tracking and model management)
- [Django](https://www.djangoproject.com/) (for running the API server)

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/breast-cancer-pipeline.git
    cd breast-cancer-pipeline
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up MLflow:**

    Make sure you have MLflow installed and running. You can start the MLflow server with:

    ```bash
    mlflow ui
    ```

    By default, MLflow runs on `http://127.0.0.1:5000`. Update the `mlflow.set_tracking_uri()` in the code if your MLflow server is running on a different address.

## Usage

### Running the Pipeline

Execute the pipeline by running the `main.py` script. The script accepts the following command-line arguments:

- `--yaml_path`: Path to the YAML configuration file.
- `--analyzer`: Optional flag to perform Exploratory Data Analysis (EDA).

To run the training process with EDA:

```bash
python main.py --yaml_path path/to/your/config.yaml --analyzer
```

To run the training process without EDA:

```bash
python main.py --yaml_path path/to/your/config.yaml
```

### Running the API Server

To start the Django API server, use the provided shell script:

```bash
./run.sh --train False
```

This will start the Django server for API operations.

### Shell Script Usage

The `run.sh` script manages both the training pipeline and the API server. It accepts the following arguments:

- `--yaml_path`: Path to the YAML configuration file (required if `--train` is `True`).
- `--analyzer`: Optional flag to perform EDA.
- `--train`: Set to `True` to run the training process or `False` to start the API server.

Example to run the training process with EDA:

```bash
./run.sh --yaml_path path/to/your/config.yaml --analyzer --train True
```

Example to start the API server:

```bash
./run.sh --train False
```

## File Structure

- `main.py`: Main script to run the pipeline.
- `run.sh`: Shell script to manage training and API server.
- `requirements.txt`: List of Python packages required for the project.
- `src/`: Source code for the pipeline components, including preprocessing, model training, and evaluation.
- `visualization/`: Contains scripts for exploratory data analysis and plotting.
- `config/`: Configuration files and schemas.

## Contributing

Contributions are welcome! To contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.