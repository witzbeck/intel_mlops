from logging import DEBUG, basicConfig, debug
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
