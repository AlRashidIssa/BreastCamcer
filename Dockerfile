# Use Arch Linux as the base image
FROM archlinux:latest

# Set the working directory
WORKDIR /BreastCancer

# Install necessary packages (Python and virtualenv)
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm python python-pip python-virtualenv gcc

# Create and activate the virtual environment, install dependencies
COPY requirements.txt /BreastCancer/
RUN python -m venv env_ && \
    /bin/bash -c "source env_/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Copy the app source code and other files to the container
COPY ./src /BreastCancer/src
COPY ./configs /BreastCancer/configs
COPY ./data /BreastCancer/data
COPY ./logs /BreastCancer/logs
COPY ./mlartifacts /BreastCancer/mlartifacts
COPY ./mlruns /BreastCancer/mlruns
COPY ./models /BreastCancer/models
COPY ./notebooks /BreastCancer/notebooks
COPY ./scripts /BreastCancer/scripts
COPY ./tests /BreastCancer/tests

# Expose the port the app runs on (API Django)
EXPOSE 8000

# Use the virtual environment to run the application
CMD ["/BreastCancer/env_/bin/python", "src/api/BreastCancerAPI/manage.py", "runserver", "0.0.0.0:8000"]
