from logging import basicConfig, debug, INFO
from os import environ, getenv
from pathlib import Path
from sys import path


debug(here := Path(__file__).parent)
# Add parent directory to path
path.append(str(Path.cwd().parent))


if not (dotenv_path := here.parent / ".env").exists():
    raise FileNotFoundError(f"Could not find {dotenv_path}")

lines = [
    x.split("#")[0].strip() if "#" in x else x
    for x in dotenv_path.read_text().splitlines()
]
pairs = [x.split("=") for x in lines if x]
for key, value in pairs:
    if key not in environ:
        debug(f"Setting {key} to {value}")
        environ[key] = value
    else:
        debug(f"{key} already set to {environ[key]}")


MODEL_NAME = getenv("MODEL")
TOKEN = getenv("HF_TOKEN")
MAX_NEW_TOKENS = int(getenv("MAX_NEW_TOKENS"))
TOP_K = int(getenv("TOP_K"))

basicConfig(level=INFO)
