# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to generate dataset for Predictive Asset Maintenance
"""

# !/usr/bin/env python
# coding: utf-8
from logging import DEBUG, Logger, basicConfig, getLogger
from pathlib import Path
from time import time
from warnings import filterwarnings

from numpy import where
from numpy.random import choice, normal, randint, seed
from pandas import DataFrame, concat, get_dummies

from __init__ import robot_dataset

basicConfig(level=DEBUG)
logger = getLogger(__name__)
filterwarnings("ignore")


DATASET_PATH = robot_dataset / "train.pkl"
MANUFACTURERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
GENERATIONS = ["Gen1", "Gen2", "Gen3", "Gen4"]
LUBRICATION_TYPES = ["LTA", "LTB", "LTC"]
PRODUCTS = ["PillA", "PillB", "PillC"]


def generate_data(
    size: int, path: Path = DATASET_PATH, random_seed: int = 1, logger: Logger = logger
) -> None:
    """
    Generate dataset for Predictive Asset Maintenance
    """
    seed(random_seed)
    # Generating our data
    start = time()
    logger.info(f"Generating data with the size {size:,}")

    data = DataFrame.from_dict(
        {
            "Age": choice(range(0, 25), size, replace=True),
            "Temperature": randint(low=50, high=300, size=size),
            "Last_Maintenance": normal(0, 60, size=size),
            "Motor_Current": randint(low=0.00, high=10.00, size=size),
            "Manufacturer": choice(MANUFACTURERS, size, replace=True),
            "Generation": choice(GENERATIONS, size, replace=True),
            "Number_Repairs": choice(range(0, 50), size, replace=True),
            "Lubrication": choice(LUBRICATION_TYPES, size, replace=True),
            "Product_Assignment": choice(PRODUCTS, size, replace=True),
        }
    )

    # Generating our target variable Asset_Label
    logger.info("Generating our target variable Asset_Label")
    data["Asset_Label"] = choice(range(0, 2), size, replace=True, p=[0.99, 0.01])

    # When age is 0-5 and over 20 change Asset_Label to 1
    logger.info("Creating correlation between our variables and our target variable")
    logger.info("When age is 0-5 and over 20 change Asset_Label to 1")
    data["Asset_Label"] = where(
        ((data.Age > 0) & (data.Age <= 5)) | (data.Age > 20), 1, data.Asset_Label
    )

    # When Temperature is between 150-300 change Asset_Label to 1
    logger.info("When Temperature is between 500-1500 change Asset_Label to 1")
    data["Asset_Label"] = where(
        (data.Temperature >= 150) & (data.Temperature <= 300), 1, data.Asset_Label
    )

    # When Manufacturer is A, E, or H change Asset_Label to have  80% 1's
    logger.info("When Manufacturer is A, E, or H change Asset_Label to 1")
    data["Temp_Var"] = choice(range(0, 2), size, replace=True, p=[0.2, 0.8])
    data["Asset_Label"] = where(
        (data.Manufacturer == "A")
        | (data.Manufacturer == "E")
        | (data.Manufacturer == "H"),
        data.Temp_Var,
        data.Asset_Label,
    )

    # When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 1's
    logger.info("When Generation is Gen1 or Gen3 change Asset_Label to have 50% to 0's")
    data["Temp_Var"] = choice(range(0, 2), size, replace=True, p=[0.5, 0.5])
    data["Asset_Label"] = where(
        (data.Generation == "Gen1") | (data.Generation == "Gen3"),
        data.Temp_Var,
        data.Asset_Label,
    )

    # When Product Assignment is Pill B change Asset_Label to have 70% to 1's
    logger.info("When District is Pill B change Asset_Label to have 70% to 1's")
    data["Temp_Var"] = choice(range(0, 2), size, replace=True, p=[0.3, 0.7])
    data["Asset_Label"] = where(
        (data.Product_Assignment == "PillB"), data.Temp_Var, data.Asset_Label
    )

    # When Lubrication is LTC change Asset_Label to have 75% to 1's
    logger.info("When Lubrication is LTC change Asset_Label to have 75% to 1's")
    data["Temp_Var"] = choice(range(0, 2), size, replace=True, p=[0.25, 0.75])
    data["Asset_Label"] = where(
        (data.Lubrication == "LTC"), data.Temp_Var, data.Asset_Label
    )

    data.drop("Temp_Var", axis=1, inplace=True)

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

    etime = time() - start
    datasize = data.shape
    logger.info(
        f"=====> Time taken {etime:2f} secs for data generation for the size of {datasize:,}"
    )

    # save data to pickle file
    logger.info(f"Saving the data to {path} ...")
    data.to_pickle(path)
    logger.info("DONE")
