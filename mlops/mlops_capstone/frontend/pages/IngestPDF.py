"""This page is the main interface for uploading and extracting output for the end user."""
import streamlit as st
from PIL import Image
from requests import post
from streamlit import (
    code,
    columns,
    divider,
    image,
    markdown,
    number_input,
    selectbox,
    slider,
    tabs,
    text_input,
    title,
)

from constants import ASSETS_PATH, STORE_PATH, TRAINING_DATA_PATH

title("Robotics Predictive Maintenance")

app_tab, help_tab = tabs(["Application", "Help"])

with app_tab:
    col11, col22 = columns(2)

    with col11:
        robot_image = Image.open(ASSETS_PATH / "robot_arm.png")
        image(robot_image)
    with col22:
        markdown(
            "##### The demand predictive asset maintenance component uses an XGBoost classifier to flag assets that need maintenance. It leverages the Intel® Extension for Scikit-Learn, XGBoost, and daal4py on Intel® 4th Generation Xeon® Scalable processors."
        )

    divider()

    markdown("#### Predictive Maintenance Model Training")

    data_file = text_input(
        "Training Data File Path",
        key="data",
        value="/home/ubuntu/certified-developer/MLOps_Professional/mlops_capstone/store/datasets/robot_maintenance/train.pkl",
    )
    model_name = text_input(
        "Model Name",
        key="model name",
        help="The name of the model without extensions",
        value="model",
    )
    model_path = text_input(
        "Model Save Path",
        key="model path",
        help="Provide the path without file name",
        value="./",
    )
    test_size = slider(
        "Percentage of data saved for Testing",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
    )
    ncpu = number_input("Threads", min_value=2, max_value=16, step=2)
    mlflow_tracking_uri = text_input(
        "Tracking URI",
        key="uri",
        value=STORE_PATH / "models/robot_maintenance",
    )
    mlflow_new_experiment = text_input("New Experiment Name", key="new exp")
    mlflow_experiment = text_input("Existing Experiment Name", key="existing exp")

    # logic for training API connections
    if st.button("Train Model", key="training"):
        URL = "http://localhost:80/train"
        DATA = {
            "file": data_file,
            "model_name": model_name,
            "model_path": model_path,
            "test_size": test_size,
            "ncpu": ncpu,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_new_experiment": mlflow_new_experiment,
            "mlflow_experiment": mlflow_experiment,
        }
        response = post(URL, json=DATA)
        if len(response.text) < 40:
            st.error(f"Model Training Failed: {response.text}")
            st.info(response.text)
        else:
            st.success(f"Model Training Completed: {response.text}")
            st.info(
                "Model Validation Accuracy Score: "
                + str(response.json()["validation_score"])
            )

    divider()

    markdown("#### Predictive Maintenance Analysis")

    model_name = text_input("Model Name", key="model name option", value="model")
    stage = manufacturer = selectbox("Model Stage", options=["Staging", "Production"])
    model_run_id = text_input("Run ID", key="model id")
    scaler_file_name = text_input(
        "Scaler File Name", key="scalar file", value="model_scaler.joblib"
    )
    scaler_destination = text_input(
        "Scaler Destination",
        key="scalerdest",
        value=TRAINING_DATA_PATH,
    )

    col21, col22, col23 = columns(3)

    manufacturer_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    model_list = ["Gen1", "Gen2", "Gen3", "Gen4"]
    lubrication_type_list = ["LTA", "LTB", "LTC"]
    product_assignment_list = ["PillA", "PillB", "PillC"]

    with col21:
        manufacturer = selectbox("Manufacturer", manufacturer_list)
        generation = selectbox("Generation", model_list)
        age = number_input("Robot Age", min_value=0, max_value=25, step=1, value=0)

    with col22:
        temperature = number_input("Temperature", min_value=50, max_value=300, step=1)
        motor_current = number_input(
            "Motor Current", min_value=0.00, max_value=10.00, step=0.05, value=5.00
        )
        lubrication_type = selectbox("Lubrication Type", lubrication_type_list)
    with col23:
        last_maintenance = number_input(
            "Last Maintenance", min_value=0, max_value=60, step=1
        )
        num_repairs = number_input("Repair Counts", min_value=0, max_value=50, step=1)
        product_assignment = selectbox(
            "Pill Product Assignment", product_assignment_list
        )

    sample = [
        {
            "Age": age,
            "Temperature": temperature,
            "Last_Maintenance": last_maintenance,
            "Motor_Current": motor_current,
            "Number_Repairs": num_repairs,
            "Manufacturer": manufacturer,
            "Generation": generation,
            "Lubrication": lubrication_type,
            "Product_Assignment": product_assignment,
        }
    ]

# logic for inference API connections

# Help tab frontend below

with help_tab:
    markdown("#### Input Descriptions:")
    markdown("- Manufacturer: Provide the name of the manufacturer of the robotic arm")
    markdown("- Model: Specify the model or specific type of the robotic arm. ")
    markdown(
        "- Lubrication Type: Indicate the type of lubrication used in the robotic arm."
    )
    markdown(
        "- Pill Type: Specify the type or category that the robotic arm is assigned to"
    )
    markdown(
        "- Age of the Machine: Enter the age or duration of use of the robotic arm."
    )
    markdown(
        "- Motor Current: Provide the current reading from the motor of the robotic arm. "
    )
    markdown(
        "- Temperature of Sensors: Specify the temperature readings from the sensors installed on the robotic arm."
    )
    markdown(
        "- Number of Historic Repairs: Enter the total number of repairs or maintenance activities performed on the robotic arm in the past. "
    )
    markdown(
        "- Last Maintenance Date: Provide the date of the last maintenance activity performed on the robotic arm."
    )
    markdown("#### Code Samples:")

    markdown("##### Conversion of XGBoost to Daal4py Model")
    daalxgboost_code = """xgb_model = xgb.train(self.parameters, xgb_train, num_boost_round=100)
        self.d4p_model = d4p.get_gbt_model_from_xgboost(xgb_model)"""
    code(daalxgboost_code, language="python")

    markdown("##### Inference with Daal4py Model")
    daalxgboost_code = """
    daal_predict_algo = d4p.gbt_classification_prediction(
            nClasses=num_class,
            resultsToEvaluate="computeClassLabels",
            fptype='float')
            
    daal_prediction = daal_predict_algo.compute(data, daal_model)
    """
    code(daalxgboost_code, language="python")

    markdown(
        "[Visit GitHub Repository for Source Code](https://github.com/intel/AI-Hackathon)"
    )
