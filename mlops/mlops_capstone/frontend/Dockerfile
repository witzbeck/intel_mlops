FROM python:3.11-slim

# set PATH
ENV PATH="/root/.local/bin:${PATH}"

# copy assets over to image
COPY . /frontend

# set the working directory
WORKDIR /frontend

# install dependancies
RUN pip3 install --user --no-cache-dir .

RUN python3 frontend/Home.py
RUN python3 pages/Monitoring.py
RUN python3 pages/RoboMaintenance.py