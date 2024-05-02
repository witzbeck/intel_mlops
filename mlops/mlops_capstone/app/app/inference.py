""" This module contains the inference function that is used to make predictions on new data samples. 
"""
from pathlib import Path
from time import time

import joblib
import mlflow
import numpy as np

from pandas import get_dummies, concat, DataFrame, merge

PROJECT_PATH = Path(__file__).parent.parent
STORE_PATH = PROJECT_PATH / "store"
OUTPUTS_PATH = STORE_PATH / "outputs"


def inference(
    model_name: str,
    stage: str,
    model_run_id: int,
    scaler_file_name: str,
    scaler_destination: str,
    data: str,
):
    # retrieve scaler
    mlflow.artifacts.download_artifacts(
        run_id=model_run_id, artifact_path=scaler_file_name, dst_path=scaler_destination
    )

    # load robust scaler
    with open(scaler_destination + f"/{scaler_file_name}", "rb") as fh:
        robust_scaler = joblib.load(fh.name)

    # load model
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

    # process data sample
    Categorical_Variables = get_dummies(
        data[["Manufacturer", "Generation", "Lubrication", "Product_Assignment"]],
        drop_first=False,
    )
    data = concat([data, Categorical_Variables], axis=1)
    data.drop(
        ["Manufacturer", "Generation", "Lubrication", "Product_Assignment"],
        axis=1,
        inplace=True,
    )

    data = data.astype({"Motor_Current": "float64", "Number_Repairs": "float64"})

    number_samples = data.select_dtypes(["float", "int", "int32"])
    scaled_samples = robust_scaler.transform(number_samples)
    scaled_samples_transformed = DataFrame(
        scaled_samples, index=number_samples.index, columns=number_samples.columns
    )
    del scaled_samples_transformed["Number_Repairs"]
    data = data.drop(
        ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
    )
    data = data.astype(int)
    processed_sample = concat([scaled_samples_transformed, data], axis=1)
    processed_sample = processed_sample.astype({"Motor_Current": "float64"})

    column_names = [
        "Age",
        "Temperature",
        "Last_Maintenance",
        "Motor_Current",
        "Number_Repairs",
        "Manufacturer_A",
        "Manufacturer_B",
        "Manufacturer_C",
        "Manufacturer_D",
        "Manufacturer_E",
        "Manufacturer_F",
        "Manufacturer_G",
        "Manufacturer_H",
        "Manufacturer_I",
        "Manufacturer_J",
        "Generation_Gen1",
        "Generation_Gen2",
        "Generation_Gen3",
        "Generation_Gen4",
        "Lubrication_LTA",
        "Lubrication_LTB",
        "Lubrication_LTC",
        "Product_Assignment_PillA",
        "Product_Assignment_PillB",
        "Product_Assignment_PillC",
    ]

    zeroes_dataframe = DataFrame(0, index=np.arange(1), columns=column_names)
    merged_df = merge(
        zeroes_dataframe,
        processed_sample,
        on=processed_sample.columns.tolist(),
        how="right",
    ).fillna(0)

    columns_to_convert = [
        "Manufacturer_A",
        "Manufacturer_B",
        "Manufacturer_C",
        "Manufacturer_D",
        "Manufacturer_E",
        "Manufacturer_F",
        "Manufacturer_G",
        "Manufacturer_H",
        "Manufacturer_I",
        "Manufacturer_J",
        "Generation_Gen1",
        "Generation_Gen2",
        "Generation_Gen3",
        "Generation_Gen4",
        "Lubrication_LTA",
        "Lubrication_LTB",
        "Lubrication_LTC",
        "Product_Assignment_PillA",
        "Product_Assignment_PillB",
        "Product_Assignment_PillC",
    ]

    merged_df[columns_to_convert] = merged_df[columns_to_convert].astype(int)

    start_time = time()
    xgb_prediction = model.predict(merged_df)
    elapsed_time_milliseconds = (time() - start_time) * 1000
    print(f"Elapsed Time: {elapsed_time_milliseconds} ms")
    for prediction in xgb_prediction:
        if prediction == 0:
            status = "Equipment Does Not Require Scheduled Maintenance"
        elif prediction == 1:
            status = "Equipment Requires Scheduled Maintenance - Plan Accordingly"

    # logic for monitoring log file creation


    return status
