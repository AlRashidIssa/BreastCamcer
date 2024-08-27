import os
import sys
import threading
from typing import Any

import numpy as np
import pandas as pd

from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required, user_passes_test
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect

sys.path.append("/app")

from src.api.BreastCancerAPI.BreastCancerAPP.dev_interface import run_main_function
from src.components.PreprocessAndPrediction import APIPredict
from src.prompt_engineer.genmi_google import BreastCancerDiagnosis
from src.utils.logging import critical
from .forms import LoginForm, RegisterForm


def main_page(request):
    """
    Render the main page.
    """
    return render(request, 'main.html')


def login_view(request):
    """
    Handle user login.
    """
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            return redirect('index')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def register_view(request):
    """
    Handle user registration.
    """
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/login')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def index(request):
    """
    Handle CSV file upload or manual input and redirect to results.
    """
    try:
        if request.method == 'POST':
            if 'csv-file' in request.FILES:
                # Handle CSV file upload
                csv_file = request.FILES['csv-file']
                if csv_file.name.endswith('.csv'):
                    fs = FileSystemStorage()
                    filename = fs.save(csv_file.name, csv_file)
                    uploaded_file_path = fs.url(filename)

                    # Process the CSV file
                    data = pd.read_csv(fs.path(filename))  # Using the actual path
                    # Perform prediction
                    predictions = predict(X=data)

                    # Store predictions in session
                    request.session['predictions'] = predictions

                    return redirect('results')  # Redirect to the results page

            elif 'input-values' in request.POST:
                # Handle input values
                input_values = request.POST.get('input-values')
                df_pre = pd.DataFrame([input_values.split(',')], columns=["id","radius_mean","texture_mean","perimeter_mean",
                                                             "area_mean","smoothness_mean","compactness_mean",
                                                             "concavity_mean","concave points_mean","symmetry_mean",
                                                             "fractal_dimension_mean","radius_se","texture_se","perimeter_se",
                                                             "area_se","smoothness_se","compactness_se","concavity_se",
                                                             "concave points_se","symmetry_se","fractal_dimension_se","radius_worst",
                                                             "texture_worst","perimeter_worst","area_worst","smoothness_worst",
                                                             "compactness_worst","concavity_worst","concave points_worst",
                                                             "symmetry_worst","fractal_dimension_worst"])
                # Perform prediction
                predictions = predict(X=df_pre)
                # Store predictions in session
                request.session['predictions'] = predictions

                return redirect('results')  # Redirect to the results page

        return render(request, 'index.html')
    except Exception as e:
        # Log the error and redirect to the error page
        critical(f"Error in index view: {str(e)}")
        return render(request, 'errors.html', {'error': 'An unexpected error occurred. Please try again later.'})


def result(request):
    """
    Display the results of the prediction.
    """
    try:
        predictions = request.session.get('predictions', None)
        if predictions is None:
            return redirect('index')

        diagnosis_tool = BreastCancerDiagnosis(predictions)
        explanation = diagnosis_tool.generate_diagnosis()

        # Ensure `explanation` is a plain string
        if isinstance(explanation, list):
            explanation = '\n'.join(explanation)

        return render(request, 'result.html', {'predictions': predictions,
                                               'massage_promet': explanation})
    except Exception as e:
        # Log the error and redirect to the error page
        critical(f"Error in result view: {str(e)}")
        return render(request, 'errors.html', {'error': 'An unexpected error occurred while processing results. Please try again later.'})


def predict(X: pd.DataFrame) -> Any:
    """
    Predict the output using a trained model.

    Args:
        model_path (str): The path to the trained model file.
        X (pd.DataFrame): The input data for which predictions are to be made.

    Returns:
        Any: The prediction results from the model.
    
    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the prediction fails.
    """
    try:
        predictions = APIPredict().call(X=X)
        return predictions
    except FileNotFoundError as e:
        # Log and re-raise the exception
        critical("Model file not found")
        raise FileNotFoundError("Model file not found") from e
    except ValueError as e:
        # Log and re-raise the exception
        critical("Error occurred during prediction")
        raise ValueError("Error occurred during prediction") from e
    except Exception as e:
        # Log and re-raise any other unexpected exceptions
        critical(f"Unexpected error during prediction: {str(e)}")
        raise



@login_required
def dev_interface(request):
    messages_output = []
    CONFIG_DIR = '/app/configs'

    if request.method == "POST":
        # Handle file upload
        config_file = request.FILES.get('config')
        if config_file:
            config_path = os.path.join(CONFIG_DIR, config_file.name)
            with open(config_path, 'wb+') as destination:
                for chunk in config_file.chunks():
                    destination.write(chunk)

            # Prepare parameters for background processing
            analyzer = 'analyzer' in request.POST
            train = 'train' in request.POST
            mlflow = 'mlflow' in request.POST
            mlflow_ui = 'mlflow-ui' in request.POST

            # Run the main function in a background thread
            def run_in_background():
                messages = run_main_function(config_path, analyzer, train, mlflow, mlflow_ui)
                # You can save messages to a file or database for retrieval
                with open('background_task_output.txt', 'a') as f:
                    for message in messages:
                        f.write(message + '\n')

            thread = threading.Thread(target=run_in_background)
            thread.start()

            # Immediate feedback
            messages_output.append('Processing has started in the background. URL for MLflow UI http://127.0.0.1:8080/')

    return render(request, 'dev_interface.html', {'messages': messages_output})