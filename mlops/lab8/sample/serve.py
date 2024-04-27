from pathlib import Path
import uvicorn
import os
import requests

from tqdm import tqdm
from langchain_community.llms import GPT4All
from fastapi import FastAPI
from model import GenPayload
from PickerBot import PickerBot
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_NAME = os.getenv("HF_MODEL_NAME")
MODEL_FILE = os.getenv("HF_FILE_URL")
MODEL_PATH = "sample/model.gguf"

if MODEL_NAME is None:
    raise ValueError("HF_MODEL environment variable not set")
if MODEL_FILE is None:
    raise ValueError("HF_FILE environment variable not set")

# app = FastAPI()


def load_gpt4allj(
    model_path: Path = MODEL_PATH,
    n_threads: int = 6,
    max_tokens: int = 50,
    repeat_penalty: float = 1.20,
    n_batch: int = 6,
    top_k: int = 1,
    device: str = "cpu",
) -> GPT4All:
    """
    # make the model path a Path object
    model_path = Path(model_path) if not isinstance(model_path, Path) else model_path
    # create the directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # resolve the path
    model_path = model_path.resolve()

    if not model_path.exists() or not model_path.is_file():
        # download model
        url = f"https://huggingface.co/{MODEL_NAME}/resolve/main/{MODEL_FILE}"
        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True)

        # open the file in binary mode and write the contents of the response to it in chunks
        # This is a large file, so be prepared to wait.
        with open(model_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=10000)):
                if chunk:
                    f.write(chunk)
    else:
        print("model already exists in path.")
    """

    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]
    # Verbose is required to pass to the callback manager
    llm = GPT4All(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        callbacks=callbacks,
        verbose=True,
        n_threads=n_threads,
        n_predict=max_tokens,
        repeat_penalty=repeat_penalty,
        n_batch=n_batch,
        top_k=top_k,
        device=device,
    )

    return llm


gptj = load_gpt4allj(
    model_path=MODEL_PATH,
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
async def predict(payload: GenPayload):
    bot = PickerBot(payload.data, model=gptj)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input=payload.user_input)
    return {"msg": "Sim Search and Inference Complete", "PickerBot Response": response}


if __name__ == "__main__":
    # uvicorn.run("serve:app", host="0.0.0.0", port=5000, log_level="info")
    bot = PickerBot("sample/data.txt", model=gptj)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input="How do I fix my apple picker?")
    print("Inference Complete!")
    print(response)