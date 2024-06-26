from dataclasses import dataclass
from json import dumps
from os import getenv


headers = {"Content-Type": "application/json"}


@dataclass(slots=True)
class DataPayload:
    """base class for data structure to send to the API"""
    endpoint: str

    @property
    def url_base(self) -> str:
        return getenv("URL_BASE")

    @property
    def url(self) -> str:
        if self.url_base is None:
            raise ValueError("URL_BASE is not set")
        if self.endpoint is None:
            raise ValueError("endpoint is not set")
        return f"{self.url_base}{self.endpoint}"

    @property
    def attrs(self) -> dict:
        return {
            k: getattr(self, k)
            for k in self.__annotations__.keys()
            if getattr(self, k) is not None
        }

    @property
    def json_str(self) -> dict:
        return f"'{dumps(self.attrs)}'"


@dataclass(slots=True)
class TrainPayload(DataPayload):
    file: str
    model_name: str
    model_path: str
    test_size: int = 25
    ncpu: int = 4
    mlflow_tracking_uri: str = None
    mlflow_new_experiment: str = None
    mlflow_experiment: str = None
    endpoint: str = "/train"


@dataclass(slots=True)
class PredictionPayload(DataPayload):
    sample: list
    model_run_id: str
    scaler_file_name: str
    scaler_destination: str = "./"
    d4p_file_name: str = None
    d4p_destination: str = None
    endpoint: str = "/predict"
