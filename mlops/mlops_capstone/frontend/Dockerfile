FROM python:3.12-slim

# set PATH
ENV PATH="/root/.local/bin:${PATH}"

# copy assets over to image
COPY . /frontend

# set the working directory
WORKDIR /frontend

# install dependancies
RUN pip3 install --user --no-cache-dir .

ARG STREAMLIT_PORT=8501

ENTRYPOINT ["streamlit", "run", "frontend/Home.py", "--server.port", $STREAMLIT_PORT, "--server.address", "0.0.0.0"]
