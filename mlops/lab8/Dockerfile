# Use an official runtime as a parent image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Copy the requirements file from the host to the container
COPY ./requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app directory contents into the container at /app
COPY . .

# Install the package
RUN pip install -e .

# Make port 80 available to the world outside this container
EXPOSE 80