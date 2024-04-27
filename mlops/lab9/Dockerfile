# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# install dependancies
RUN apt-get update && apt-get install -y \
python3-dev \
build-essential \
libpq-dev \
&& rm -rf /var/lib/apt/lists/*

COPY . .

# set PATH
ENV PATH=/root/.local/bin:$PATH

# Install Jupyter Kernel Gateway
RUN pip3 install -U --no-cache-dir jupyter_kernel_gateway

RUN pip3 install -U --upgrade pip

# Install the required libraries for your notebook
RUN pip3 install -U --no-cache-dir -e .

# Expose the port used by Kernel Gateway
EXPOSE 8888

ENV JUPYTER_GATEWAY_IP \
    JUPYTER_GATEWAY_PORT \
    JUPYTER_GATEWAY_API \
    JUPYTER_ALLOW_ORIGIN \
    JUPYTER_ALLOW_HEADERS \
    JUPYTER_ALLOW_METHODS

# Run Jupyter Kernel Gateway
CMD ["sh", "-c", "jupyter-kernel-gateway --ip=${JUPYTER_GATEWAY_IP} --port=${JUPYTER_GATEWAY_PORT} --KernelGatewayApp.api='${JUPYTER_GATEWAY_API}' --KernelGatewayApp.allow_origin='${JUPYTER_ALLOW_ORIGIN}' --KernelGatewayApp.allow_headers='${JUPYTER_ALLOW_HEADERS}' --KernelGatewayApp.allow_methods='${JUPYTER_ALLOW_METHODS}'"]
