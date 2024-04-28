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

from utils import generate_data

if __name__ == "__main__":
    basicConfig(level=DEBUG)
    logger = getLogger(__name__)
    filterwarnings("ignore")

    parser = ArgumentParser()
    parser.add_argument(
        "--generate", type="store_true", required=False, help="generate data"
    )
    parser.add_argument(
        "-s", "--size", type=int, required=False, default=25000, help="data size"
    )
    FLAGS = parser.parse_args()

    if FLAGS.generate:
        generate_data(FLAGS.size)
    else:
        raise ValueError("Unrecognized option")
