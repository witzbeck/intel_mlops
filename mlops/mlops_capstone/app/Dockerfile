FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

# set PATH
ENV PATH="/root/.local/bin:${PATH}"

# copy assets over to image
COPY . /app

# set the working directory
WORKDIR /app

# install dependancies
RUN pip3 install --user --no-cache-dir -e .

# generate training data
RUN python -m app --generate -s 50000
