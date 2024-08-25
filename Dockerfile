# Use Arch Linux as the base image
FROM archlinux:latest

# Set the working directory
WORKDIR /app

# Install necssary packages (Python and virtualenv)
RUN pacman -Suy --noconfirm && pacman -S --noconfirm python python-pip python-virtualenv gcc

# Create the virtual environment named 'env_'
RUN python -m venv env_

# Activate the virtual environment and install dependncies
RUN /bis/bash -c "source env_/bin/activate && pip install --no-cache-dir -r  /app/requirements.txt"

# Copy the app source code to the container
COPY  ./src /app
COPY  ./config /app
COPY  ./data /app
COPY  ./logs /app
COPY  ./mlartifacts /app
COPY  ./mlruns /app
COPY  ./models /app
COPY  ./notebooks /app
COPY  ./scripts /app
COPY  ./test /app
COPY  ./requirements.txt /app

# Expose the port the app runs on (API Django)
EXPOSE 8000 

# Use the virtual envri to run the application
CMD ["/bin/bash", "-c", "source env_/bin/activate && python src/api/BreastCancerAPI/manage.py runservert 0.0.0.0.8000"]
