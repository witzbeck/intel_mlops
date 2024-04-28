from pathlib import Path

here = Path(__file__).parent
store = here.parent / "store"
print(f"Store path: {store}")

(models := store / "models").mkdir(exist_ok=True, parents=True)
(datasets := store / "datasets").mkdir(exist_ok=True, parents=True)
(outputs := store / "outputs").mkdir(exist_ok=True, parents=True)

(robot_dataset := datasets / "robot_maintenance").mkdir(exist_ok=True, parents=True)
