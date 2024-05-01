FROM python:3.12-slim

# set PATH
ENV PATH="/root/.local/bin:${PATH}"

# copy assets over to image
COPY . /frontend

# set the working directory
WORKDIR /frontend

# install dependancies
RUN pip3 install --user --no-cache-dir .
