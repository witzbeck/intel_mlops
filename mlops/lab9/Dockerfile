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

RUN pip3 install --upgrade pip

# Install the required libraries for your notebook
RUN pip3 install -U --no-cache-dir -r requirements.txt

# Expose the port used by Kernel Gateway
EXPOSE 8888

# Run Jupyter Kernel Gateway
CMD ["jupyter-kernel-gateway", "--ip=0.0.0.0", "--KernelGatewayApp.api='kernel_gateway.notebook_http'", "--KernelGatewayApp.allow_origin='*'", "--KernelGatewayApp.allow_headers='Content-Type'", "--KernelGatewayApp.allow_methods='GET, POST, DELETE, OPTIONS'"]
