from mlflow import MlflowClient

client = MlflowClient()

# List all registered models
registered_models = client.get_registered_model(name="GradientBoostingModel")

# Iterate over each registered model and list its versions
for model in registered_models:
    model_name = model.name
    # List versions for each model
    versions = client.list_model_versions(model_name)
    version_list = [v.version for v in versions]
    print(f"Model Name: {model_name}, Versions: {version_list}")
