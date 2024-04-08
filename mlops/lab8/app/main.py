from logging import debug
from os import getenv
from pathlib import Path

from fastapi import FastAPI
from langchain.llms.gpt4all import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tqdm import tqdm
from requests import get
from transformers import AutoModelForCausalLM, AutoTokenizer
from uvicorn import run

from model.data_model import GenPayload
from model.picker_bot import PickerBot

app = FastAPI()



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
