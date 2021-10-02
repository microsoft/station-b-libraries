# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from flask import Flask

app = Flask(__name__)
app.config["UPLOADS_DIR"] = "./uploads"
from api import routes  # type: ignore  # noqa: E402

assert routes is not None  # for flake8
