# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gp",
    version="0.0.1",
    author="",
    author_email="",
    description="Global Penalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=["gp"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "torch",
    ],
)
