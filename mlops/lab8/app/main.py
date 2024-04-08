from logging import debug
from os import getenv
from pathlib import Path

from requests import get
from uvicorn import run
from tqdm import tqdm
from langchain.llms.gpt4all import GPT4All
from fastapi import FastAPI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from model.data_model import GenPayload
from model.picker_bot import PickerBot

app = FastAPI()


def load_gpt4allj(
    model_path: Path,
    n_threads: int = 6,
    max_tokens: int = 50,
    repeat_penalty: float = 1.20,
    n_batch: int = 6,
    top_k: int = 1,
) -> GPT4All:
    """Load the GPT4All model"""
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if not model_path.is_file():
        # download model
        url = getenv("MODEL_URL")
        # send a GET request to the URL to download the file. Stream since it's large
        response = get(url, stream=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # open the file in binary mode and write the contents of the response to it in chunks
        # This is a large file, so be prepared to wait.
        with open(model_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=10000)):
                if chunk:
                    f.write(chunk)
    elif not model_path.exists():
        raise FileNotFoundError("model does not exist in path.")
    else:
        debug("model already exists in path.")

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(
        model=model_path,
        callbacks=callbacks,
        verbose=True,
        n_threads=n_threads,
        n_predict=max_tokens,
        repeat_penalty=repeat_penalty,
        n_batch=n_batch,
        top_k=top_k,
    )

    return llm


gptj = load_gpt4allj(
    model_path=getenv("MODEL_PATH"),
    n_threads=15,
    max_tokens=100,
    repeat_penalty=1.20,
    n_batch=15,
    top_k=1,
)


@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message": "Server is Running"}


@app.post("/predict")
async def predict(payload: GenPayload) -> dict[str: str]:
    """Endpoint to perform similarity search and inference"""
    bot = PickerBot(payload.data, model=gptj)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input=payload.user_input)
    return {"message": "Sim Search and Inference Complete", "PickerBot Response": response}


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=80, log_level="debug")
