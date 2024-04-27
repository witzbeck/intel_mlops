from pathlib import Path
from os import getenv

from langchain_community.llms import GPT4All
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from PickerBot import PickerBot

MODEL_NAME = getenv("HF_MODEL_NAME")
MODEL_FILE = getenv("HF_FILE_URL")
MODEL_PATH = "model.gguf"

if MODEL_NAME is None:
    raise ValueError("HF_MODEL environment variable not set")
if MODEL_FILE is None:
    raise ValueError("HF_FILE environment variable not set")


# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]


# Verbose is required to pass to the callback manager
model = GPT4All(
    model=MODEL_PATH,
    model_name=MODEL_NAME,
    repeat_penalty=1.20,
    callbacks=callbacks,
    n_batch=6,
    top_k=1,
    device="cpu",
    n_threads=15,
    verbose=True,
    max_tokens=100,
)


if __name__ == "__main__":
    bot = PickerBot("sample/data.txt", model=model)
    bot.data_proc()
    bot.create_vectorstore()
    response = bot.inference(user_input="How do I fix my apple picker?")
    print("Inference Complete!")
    print(response)
