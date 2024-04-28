from os import environ
from pathlib import Path

here = Path(__file__).parent
dotenv_file = here / ".env"
if not dotenv_file.exists():
    raise FileNotFoundError(f"Missing .env file at {dotenv_file}")

env_lines = [
    line.strip()
    for line in dotenv_file.read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]
pairs = {
    k: v for k, v in (line.split("=", 1) for line in env_lines)
}
environ.update(pairs)

store = here / "store"

(models := store / "models").mkdir(exist_ok=True, parents=True)
(datasets := store / "datasets").mkdir(exist_ok=True, parents=True)
(outputs := store / "outputs").mkdir(exist_ok=True, parents=True)

(robot_dataset := datasets / "robot_maintenance").mkdir(exist_ok=True, parents=True)
