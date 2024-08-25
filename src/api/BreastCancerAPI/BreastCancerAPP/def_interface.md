Each scenario represents a different combination of options (`--config`, `--analyzer`, `--train`, `--mlflow`) that you can use to control the script's behavior.



### 2. **Running the Training Pipeline Only**
This scenario runs the training pipeline using the provided configuration file. No MLflow tracking is involved.

**Command:**
```bash
python dev_interface.py --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml --train
```

**Expected Output:**
```text
Changing directory to the BreastCancer project directory...
Changed directory to /home/alrashidissa/Desktop/BreastCancer
Virtual environment activated
Starting training process...
Changed directory to /home/alrashidissa/Desktop/BreastCancer/src/training
Training pipeline started with the provided config file...
```

### 3. **Running the Training Pipeline with Analyzer**
In this scenario, the training pipeline is run with the analyzer enabled. This could involve additional analysis steps during training.

**Command:**
```bash
python dev_interface.py --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml --train --analyzer
```

**Expected Output:**
```text
Changing directory to the BreastCancer project directory...
Changed directory to /home/alrashidissa/Desktop/BreastCancer
Virtual environment activated
Starting training process...
Changed directory to /home/alrashidissa/Desktop/BreastCancer/src/training
Training pipeline started with the provided config file...
Analyzer enabled...
```

### 4. **Running the MLflow Pipeline Only**
This scenario runs the MLflow pipeline using the provided configuration file, without triggering any training processes.

**Command:**
```bash
python dev_interface.py --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml --mlflow
```

**Expected Output:**
```text
Changing directory to the BreastCancer project directory...
Changed directory to /home/alrashidissa/Desktop/BreastCancer
Virtual environment activated
MLflow configuration is enabled. Starting MLflow pipeline...
Changed directory to /home/alrashidissa/Desktop/BreastCancer/src/mlflow
MLflow pipeline started with the provided config file...
```





**Expected Output:**
```text
Changing directory to the BreastCancer project directory...
Changed directory to /home/alrashidissa/Desktop/BreastCancer
Virtual environment activated
Error: --config is required when --train is True.
```

### 7. **Error Scenario: Invalid Directory**
This scenario illustrates what happens if the script fails to change to the specified project directory.

**Command:**
```bash
python dev_interface.py --config /home/alrashidissa/Desktop/BreastCancer/configs/config.yaml
```

If the directory does not exist, the expected output would be:

```text
Changing directory to the BreastCancer project directory...
Error: Failed to change directory to /home/alrashidissa/Desktop/BreastCancer
Error: Failed to activate the virtual environment
No training or MLflow specified. Starting Django API server...
```

### Summary

- **Basic API Server**: Run the Django API server without any training or MLflow (`--config` only).
- **Training Only**: Run the training pipeline with the provided config (`--config --train`).
- **Training with Analyzer**: Run the training pipeline with the analyzer (`--config --train --analyzer`).
- **MLflow Only**: Run the MLflow pipeline (`--config --mlflow`).
- **Training and MLflow**: Run both the training and MLflow pipelines (`--config --train --mlflow`).
- **Error Handling**: Demonstrates error scenarios when required parameters are missing or directories are invalid.


