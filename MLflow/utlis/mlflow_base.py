import MLflow
from typing import Any, Optional, Dict

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags: Dict[str, Any]) -> str:
    """
    Create a new MLflow experiment.

    Parameters:
    -----------
    experiment_name : str
        The name of the experiment.
    artifact_location : str
        The artifact location where MLflow will store the experiment data.
    tags : dict
        Tags associated with the experiment.

    Returns:
    --------
    str
        The ID of the created experiment.
    """
    try:
        experiment_id = MLflow.create_experiment( # type: ignore
            experiment_name,
            artifact_location=artifact_location,
            tags=tags
        )
    except MLflow.exceptions.MlflowException: # type: ignore
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = MLflow.get_experiment_by_name(experiment_name).experiment_id # type: ignore
    
    return experiment_id

def get_mlflow_experiment(experiment_id: Optional[str] = None, experiment_name: Optional[str] = None) -> MLflow.entities.Experiment: # type: ignore
    """
    Retrieve an MLflow experiment.

    Parameters:
    -----------
    experiment_id : str, optional
        The ID of the experiment to retrieve.
    experiment_name : str, optional
        The name of the experiment to retrieve.

    Returns:
    --------
    mlflow.entities.Experiment
        The MLflow experiment.
    """
    if experiment_id is not None:
        experiment = MLflow.get_experiment(experiment_id) # type: ignore
    elif experiment_name is not None:
        experiment = MLflow.get_experiment_by_name(experiment_name) # type: ignore
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    
    return experiment

def delete_mlflow_experiment(experiment_id: Optional[str] = None, experiment_name: Optional[str] = None) -> None:
    """
    Delete an MLflow experiment.

    Parameters:
    -----------
    experiment_id : str, optional
        The ID of the experiment to delete.
    experiment_name : str, optional
        The name of the experiment to delete.
    """
    if experiment_id is not None:
        MLflow.delete_experiment(experiment_id) # type: ignore
    elif experiment_name is not None:
        experiment = MLflow.get_experiment_by_name(experiment_name) # type: ignore
        experiment_id = experiment.experiment_id  # type: ignore
        MLflow.delete_experiment(experiment_id) # type: ignore
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
