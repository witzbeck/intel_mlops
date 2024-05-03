"""
Module to train and prediction using XGBoost Classifier
"""
# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error

from logging import DEBUG, basicConfig, getLogger
from os import getenv
from pathlib import Path
from sys import exit
from warnings import filterwarnings

from joblib import dump
from mlflow import (
    create_experiment,
    get_experiment_by_name,
    log_artifact,
    log_metric,
    search_runs,
    set_experiment,
    set_tracking_uri,
    start_run,
    xgboost,
)
from numpy import array, count_nonzero, ravel
from pandas import DataFrame, concat, read_pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import DMatrix, train

STORE_PATH = Path("/app/store")
OUTPUTS_PATH = STORE_PATH / "outputs"
TRAINING_DATA_PATH = STORE_PATH / "datasets/robot_maintenance/train.pkl"
tracking_uri = getenv("MLFLOW_TRACKING_URI")

basicConfig(level=DEBUG)
logger = getLogger(__name__)
filterwarnings("ignore")


class RoboMaintenance:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.file = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled_transformed = None
        self.X_test_scaled_transformed = None
        self.accuracy_scr = None
        self.model_path = None
        self.parameters = None
        self.robust_scaler = None
        self.run_id = None
        self.active_experiment = None
        self.xgb_model = None

    def mlflow_tracking(
        self,
        tracking_uri: str = tracking_uri,
        experiment: str = None,
        new_experiment: str = None,
    ):
        # sets tracking URI
        set_tracking_uri(tracking_uri)

        # creates new experiment if no experiment is specified
        if experiment is None:
            create_experiment(new_experiment)
            self.active_experiment = new_experiment
            set_experiment(new_experiment)
        else:
            set_experiment(experiment)
            self.active_experiment = experiment

    def process_data(self, file: str, test_size: int = 0.25):
        """processes raw data for training

        Parameters
        ----------
        file : str
            path to raw training data
        test_size : int, optional
            percentage of data reserved for testing, by default .25
        """

        # Generating our data
        logger.info("Reading the dataset from %s...", file)
        try:
            data = read_pickle(file)
        except FileNotFoundError:
            exit("Dataset file not found")

        X = data.drop("Asset_Label", axis=1)
        y = data.Asset_Label

        X_train, X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size
        )

        df_num_train = X_train.select_dtypes(["float", "int", "int32"])
        df_num_test = X_test.select_dtypes(["float", "int", "int32"])
        self.robust_scaler = RobustScaler()
        X_train_scaled = self.robust_scaler.fit_transform(df_num_train)
        X_test_scaled = self.robust_scaler.transform(df_num_test)

        # Making them pandas dataframes
        X_train_scaled_transformed = DataFrame(
            X_train_scaled, index=df_num_train.index, columns=df_num_train.columns
        )
        X_test_scaled_transformed = DataFrame(
            X_test_scaled, index=df_num_test.index, columns=df_num_test.columns
        )

        del X_train_scaled_transformed["Number_Repairs"]

        del X_test_scaled_transformed["Number_Repairs"]

        # Dropping the unscaled numerical columns
        X_train = X_train.drop(
            ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
        )
        X_test = X_test.drop(
            ["Age", "Temperature", "Last_Maintenance", "Motor_Current"], axis=1
        )

        X_train = X_train.astype(int)
        X_test = X_test.astype(int)

        # Creating train and test data with scaled numerical columns
        X_train_scaled_transformed = concat(
            [X_train_scaled_transformed, X_train], axis=1
        )
        X_test_scaled_transformed = concat([X_test_scaled_transformed, X_test], axis=1)

        self.X_train_scaled_transformed = X_train_scaled_transformed.astype(
            {"Motor_Current": "float64"}
        )
        self.X_test_scaled_transformed = X_test_scaled_transformed.astype(
            {"Motor_Current": "float64"}
        )

    def train(self, ncpu: int = 4):
        """trains an XGBoost Classifier and Tracks Models with MLFlow

        Parameters
        ----------
        ncpu : int, optional
            number of CPU threads used for training, by default 4
        """

        # Set xgboost parameters
        self.parameters = {
            "max_bin": 256,
            "scale_pos_weight": 2,
            "lambda_l2": 1,
            "alpha": 0.9,
            "max_depth": 8,
            "num_leaves": 2**8,
            "verbosity": 0,
            "objective": "multi:softmax",
            "learning_rate": 0.3,
            "num_class": 3,
            "nthread": ncpu,
        }

        xgboost.autolog()
        xgb_train = DMatrix(self.X_train_scaled_transformed, label=array(self.y_train))
        self.xgb_model = train(self.parameters, xgb_train, num_boost_round=100)

        # store run id for user in other methods
        xp = get_experiment_by_name(self.active_experiment)._experiment_id
        self.run_id = search_runs(xp, output_format="list")[0].info.run_id

    def validate(self):
        """performs model validation with testing data

        Returns
        -------
        float
            accuracy metric
        """

        # calculate accuracy
        dtest = DMatrix(self.X_test_scaled_transformed, self.y_test)
        xgb_prediction = self.xgb_model.predict(dtest)
        xgb_errors_count = count_nonzero(xgb_prediction - ravel(self.y_test))
        self.accuracy_scr = 1 - xgb_errors_count / xgb_prediction.shape[0]

        # log accuracy metric with mlflow
        with start_run(self.run_id):
            log_metric("accuracy", self.accuracy_scr)

        return self.accuracy_scr

    def save(self, model_path):
        """Logs scaler as mlflow artifact.

        Parameters
        ----------
        model_path : str
            path where trained model should be saved
        """

        self.scaler_path = model_path + self.model_name + "_scaler.joblib"

        logger.info("Saving Scaler")
        with open(self.scaler_path, "wb") as fh:
            dump(self.robust_scaler, fh.name)

        logger.info("Saving Scaler as MLFLow Artifact")
        with start_run(self.run_id):
            log_artifact(self.scaler_path)
