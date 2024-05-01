# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Module to generate dataset for Predictive Asset Maintenance
"""

# !/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from logging import DEBUG, basicConfig, getLogger
from warnings import filterwarnings

from app.utils import generate_data
from requests import get

url_base = "http://127.0.0.1:80"
headers = {"Content-Type": "application/json"}


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    logger = getLogger(__name__)
    filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="Ping the server")
    parser.add_argument(
        "--generate", action="store_true", required=False, help="generate data"
    )
    parser.add_argument(
        "-s", "--size", type=int, required=False, default=25000, help="data size"
    )
    FLAGS = parser.parse_args()

    if FLAGS.ping:
        response = get(f"{url_base}/ping")
        if response.status_code == 200:
            print("Server is Running")
        else:
            print("Server is not Running")
    elif FLAGS.generate:
        generate_data(FLAGS.size)
    else:
        raise ValueError(f"No valid arguments passed | {parser.print_help()}")
