# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""A basic logging capability.

TODO: It seems to be not extensively used at the moment.

Exports:
    logger, the default logger of the pipeline
"""
import datetime
import logging
import sys


def setup_logger() -> logging.Logger:
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.FileHandler(f"{datetime.date.today()}.log", mode="a")
    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    default_logger = logging.getLogger("default")
    default_logger.setLevel(logging.DEBUG)
    default_logger.addHandler(handler)
    default_logger.addHandler(screen_handler)
    return default_logger


logger = setup_logger()
