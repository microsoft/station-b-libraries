# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""Setup for the staticchar package"""
import setuptools
from pkg_resources import parse_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = [str(req) for req in parse_requirements(fh)]

setuptools.setup(
    name="staticchar",
    version="0.0.1",
    author="Station B",
    author_email="",
    description="Static Characterization of fluorescence time series data.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="",
    packages=["staticchar"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
)
