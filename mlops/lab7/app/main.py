from logging import getLogger
from typing import Any

from fastapi import FastAPI
from pandas import json_normalize
from uvicorn import run

from app.data_model import TrainPayload, PredictionPayload
from app.inference import inference
from app.train import HarvesterMaintenance


app = FastAPI()
logger = getLogger(__name__)


@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message": "Server is Running"}


@app.post("/train")
async def train(payload: TrainPayload) -> dict[str, Any]:
    """Training Endpoint
    This endpoint process raw data and trains an XGBoost Classifier

    Parameters
    ----------
    payload : TrainPayload
        Training endpoint payload model

    Returns
    -------
    dict
        Accuracy metrics and other logger feedback on training progress.
    """
    model = HarvesterMaintenance(payload.model_name)
    model.mlflow_tracking(
        tracking_uri=payload.mlflow_tracking_uri,
        new_experiment=payload.mlflow_new_experiment,
        experiment=payload.mlflow_experiment,
    )
    logger.info("Configured Experiment and Tracking URI for MLFlow")
    model.process_data(payload.file, payload.test_size)
    logger.info("Data has been successfully processed")
    model.train(payload.ncpu)
    logger.info("Maintenance Apple Harvester Model Successfully Trained")
    model.save(payload.model_path)
    logger.info("Saved Harvester Maintenance Model")
    accuracy_score = model.validate()
    return {"msg": "Model trained succesfully", "validation scores": accuracy_score}


@app.post("/predict")
async def predict(payload: PredictionPayload) -> dict[str, Any]:
    sample = json_normalize(payload.sample)
    results = inference(
        model_run_id=payload.model_run_id,
        scaler_file_name=payload.scaler_file_name,
        scaler_destination=payload.scaler_destination,
        d4p_file_name=payload.d4p_file_name,
        d4p_destination=payload.d4p_destination,
        data=sample,
    )
    return {"msg": "Completed Analysis", "Maintenance Recommendation": results}


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8000, log_level="info")
