# Use Arch Linux as the base image
FROM archlinux:latest

# Set the working directory
WORKDIR /app

# Install necessary packages (Python and virtualenv)
RUN pacman -Syu --noconfirm && pacman -S --noconfirm python python-pip python-virtualenv gcc

# Create the virtual environment named 'env_'
RUN python -m venv env_

# Activate the virtual environment and install dependencies
COPY requirements.txt /app/
RUN /bin/bash -c "source env_/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Copy the app source code and other files to the container
COPY ./src /app/src
COPY ./configs /app/config
COPY ./data /app/data
COPY ./logs /app/logs
COPY ./mlartifacts /app/mlartifacts
COPY ./mlruns /app/mlruns
COPY ./models /app/models
COPY ./notebooks /app/notebooks
COPY ./scripts /app/scripts
COPY ./test /app/test

# Expose the port the app runs on (API Django)
EXPOSE 8000

# Use the virtual environment to run the application
CMD ["/bin/bash", "-c", "source env_/bin/activate && python src/api/BreastCancerAPI/manage.py runserver 0.0.0.0:8000"]
