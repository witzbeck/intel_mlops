# Use an official runtime as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Copy the requirements file from the host to the container
COPY ./requirements.txt /app/requirements.txt

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN python3 -m pip install -r ./requirements.txt

# Copy the app directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80