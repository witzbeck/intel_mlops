from logging import DEBUG, basicConfig, debug
from os import environ
from pathlib import Path
from sys import path

from warnings import filterwarnings

filterwarnings("ignore")

debug(here := Path(__file__).parent)
# Add parent directory to path
path.append(str(Path.cwd().parent))
logconfig = basicConfig(
    level=DEBUG, filename=str(here.parent / f"{here.parent.stem}.log")
)

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
