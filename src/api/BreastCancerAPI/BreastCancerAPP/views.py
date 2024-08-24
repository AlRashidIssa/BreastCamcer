from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
from typing import Any

from .forms import LoginForm, RegisterForm

import sys
sys.path.append("/home/alrashidissa/Desktop/BreastCancer")
from PrometEnginer.genmi_google import BreastCancerDiagnosis
from src.components.PreprocessAndPrediction import APIPredict
from src import BrestCancer_critical

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
                    predictions = predict(model_path='/home/alrashidissa/Desktop/BreastCancer/PreTrainModel/GradientBoostingModel.pkl', 
                                          X=data)

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
                predictions = predict(model_path='/path/to/model', X=df_pre)
                
                # Store predictions in session
                request.session['predictions'] = predictions

                return redirect('results')  # Redirect to the results page

        return render(request, 'index.html')
    except Exception as e:
        # Log the error and redirect to the error page
        BrestCancer_critical(f"Error in index view: {str(e)}")
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
        BrestCancer_critical(f"Error in result view: {str(e)}")
        return render(request, 'errors.html', {'error': 'An unexpected error occurred while processing results. Please try again later.'})


def predict(model_path: str, X: pd.DataFrame) -> Any:
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
        predictions = APIPredict().call(model_path=model_path, X=X)
        return predictions
    except FileNotFoundError as e:
        # Log and re-raise the exception
        BrestCancer_critical(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}") from e
    except ValueError as e:
        # Log and re-raise the exception
        BrestCancer_critical("Error occurred during prediction")
        raise ValueError("Error occurred during prediction") from e
    except Exception as e:
        # Log and re-raise any other unexpected exceptions
        BrestCancer_critical(f"Unexpected error during prediction: {str(e)}")
        raise
