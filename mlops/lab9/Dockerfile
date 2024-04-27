# Use an official Python runtime as a parent image
FROM jupyter/base-notebook:latest

# install dependancies
# RUN apt-get update && apt-get install -y \
# python3-dev \
# build-essential \
# libpq-dev \
# && rm -rf /var/lib/apt/lists/*

# set PATH
ENV PATH=/root/.local/bin:$PATH

COPY . /home/jovyan/work

WORKDIR /home/jovyan/work

# Install the required libraries for your notebook
RUN pip3 install -U --no-cache-dir -e .

# Expose the port used by Kernel
EXPOSE 8888
