FROM python:3.12-slim-bookworm

# install dependancies
RUN apt-get update && apt-get install -y \
python3-dev \
build-essential \
curl git git-lfs \
libpq-dev \
&& rm -rf /var/lib/apt/lists/* \
&& apt-get clean

# copy assets over to image
COPY . /pickerbot

# set the working directory
WORKDIR /pickerbot

# set PATH
ENV PATH=/root/.local/bin:$PATH


# upgrade pip
RUN pip3 install --upgrade pip

RUN ls -la && sleep 1

WORKDIR /pickerbot/sample

# install python dependancies    
RUN pip3 install --user --no-cache-dir -r requirements.txt

# ARGs for downloading the model
ARG HF_FILE_URL
ARG HF_TOKEN

# Download the model using the token in the header
RUN curl -H "Authorization: Bearer $HF_TOKEN" \
    "$HF_FILE_URL?raw=true" -o model.gguf

COPY model.gguf /pickerbot/sample/model.gguf

RUN ls -la && sleep 1

RUN pip3 install --user --no-cache-dir -e ..

# exposing endpoint port
EXPOSE 5000

ENTRYPOINT ["python", "serve.py"]
