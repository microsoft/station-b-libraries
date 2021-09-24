# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypeVar

import yaml
from azure.cosmosdb.table.tableservice import TableService
from azure.storage.blob import BlobClient
from flask import Flask, Response, jsonify, request
from werkzeug.http import HTTP_STATUS_CODES

ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))
PYBCKG_DIR = ROOT_DIR / "libraries" / "PyBCKG"
sys.path.append(str(PYBCKG_DIR))

SPECS_DIR = ROOT_DIR / "libraries/ABEX/tests/data/specs"

from abex.optimizers.optimizer_base import OptimizerBase  # noqa: E402
from abex.settings import load_config_from_path_or_name, load_resolutions  # noqa: E402  # type: ignore # auto

from libraries.PyBCKG.pyBCKG.azurestorage.api import from_connection_string  # noqa: E402
from libraries.PyBCKG.pyBCKG.utils import HttpRequestMethod  # noqa: E402

app = Flask(__name__)


@app.route("/")
@app.route("/get-experiment-options", methods=["GET"])
def get_experiments():
    connection_string = request.headers.get("storageConnectionString")

    az_conn = from_connection_string(connection_string)
    query = "experiments()"
    # queryfilter = az_conn._queryfilter('deprecated', "False")
    # queryfilter = f"?$filter=Deprecated eq '{False}'"
    queryfilter = ""
    expt_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)

    experiments = expt_json["value"]
    return Response(json.dumps(experiments), mimetype="application/json")


@app.route("/get-aml-runs", methods=["GET"])
def get_aml_runs():
    connection_string = request.headers.get("storageConnectionString")

    az_conn = from_connection_string(connection_string)
    query = "azuremlruns()"
    queryfilter = ""
    aml_runs_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)

    aml_runs = aml_runs_json["value"]

    return Response(json.dumps(aml_runs), mimetype="application/json")


@app.route("/get-experiment-result", methods=["GET"])
def get_experiment_results():
    # connection_string = request.headers.get("storageConnectionString")

    experiment_name = request.args.get("experimentName")
    print(f"experiment name: {experiment_name}")

    # TODO: Download AML run and construct IExperimentResult object

    experiment_results = [
        {
            "id": 1,
            "description": "",
            "samples": [{}],
            "signals": [{}],
            "type": "",
            "timestamp": "",
            "deprecated": "",
            "name": "experiment1",
            "iterations": ["1"],
            "folds": ["1", "2", "3"],
            "imageFolders": ["/abex-results/tutorial-intro"],
            "imageNames": [
                "slice1d_",
                "slice2d_",
                "acquisition1d_",
                "train_test_",
                "bo_distance",
                "bo_experiment",
            ],
            "suggestedExperiments": [
                {"x": 5.0, "y": 8.4},
                {"x": 5.6, "y": 8.3},
                {"x": 5.3, "y": 8.5},
                {"x": 5.7, "y": 8.8},
                {"x": 5.4, "y": 8.2},
            ],
        },
        {
            "id": 2,
            "description": "",
            "samples": [{}],
            "signals": [{}],
            "type": "",
            "timestamp": "",
            "deprecated": "",
            "name": "experiment2",
            "iterations": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "folds": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "imageFolders": ["/abex-results/synthetic_3input_batch5"],
            "imageNames": [
                "slice1d_",
                "slice2d_",
                "acquisition1d_",
                "train_only",
                "bo_distance",
                "bo_experiment",
            ],
            "suggestedExperiments": [
                {"ARA": 1.0, "ATC": 3.3, "C_on": 1},
                {"ARA": 1.1, "ATC": 3.5, "C_on": 1},
                {"ARA": 1.1, "ATC": 3.6, "C_on": 1},
                {"ARA": 1.1, "ATC": 3.8, "C_on": 1},
                {"ARA": 1.1, "ATC": 3.5, "C_on": 1},
            ],
        },
    ]

    # This will be the real result once we are storing these storing results in BCKG
    # expt_result =[ex for ex in experiment_results if ex.name == experiment_name][0]
    expt_result = experiment_results[0]

    return Response(json.dumps(expt_result), mimetype="application/json")


@app.route("/get-config-options", methods=["GET"])
def get_configs():
    configs = [{"id": "1", "name": "config1"}, {"id": "2", "name": "config2"}]

    return Response(json.dumps(configs), mimetype="application/json")


@app.route("/get-dataset-options", methods=["GET"])
def get_datasets():
    datasets = [
        {
            "id": "1",
            "name": "dataset1",
            "dateCreated": "17 Feb 2021",
            "dataRecords": [
                {
                    "SampleId": "S1",
                    "ObservationId": "O3",
                    "Arabinose": 4.67,
                    "C6": 345.23,
                    "C12": 12334.34,
                    "EYFP": 187982.23,
                    "ECFP": 23445.4,
                    "MRFP": 765.67,
                },
                {
                    "SampleId": "S2",
                    "ObservationId": "O1",
                    "Arabinose": 6.54,
                    "C6": 234.63,
                    "C12": 3243.98,
                    "EYFP": 87668.34,
                    "ECFP": 72726.21,
                    "MRFP": 7725.43,
                },
            ],
        },
        {
            "id": "2",
            "name": "dataset2",
            "dateCreated": "01 March 2021",
            "dataRecords": [
                {"A": "a", "B": 2, "C": "e", "D": 4},
                {"A": "b", "B": 1, "C": "f", "D": 1},
            ],
        },
    ]
    return Response(json.dumps(datasets), mimetype="application/json")


def parse_binary_to_dict(data: bytes) -> yaml.YAMLObject:
    header_dict = json.loads(data)
    data_dict = header_dict.get("headers")

    return data_dict


def get_file_name(data_dict: Dict[str, Any]):
    config_name = data_dict.get("fileName")
    return config_name


def get_config_data(data_dict: Dict[str, Any]):
    config_data = data_dict.get("config")
    config_json = yaml.safe_load(config_data)  # type: ignore
    return config_json


T = TypeVar("T")


def get_csv_data(data_dict: Dict[str, T]) -> T:
    observation_data = data_dict.get("observations")
    print("observation data: ", observation_data)
    # TODO: PARSE CSV
    observations = observation_data
    assert observations is not None  # since non-None return is assumed by some callers
    return observations


def get_connection_str_from_binary(data: bytes):
    header_dict = json.loads(data)
    data_dict = header_dict.get("headers")
    connection_str = data_dict.get("storageConnectionString")
    return connection_str


def http_response(status_code, message, error_code=None, response=None, reason=None):
    response_status_code = HTTP_STATUS_CODES.get(status_code, "Unknown error")
    if response_status_code == 409 and "blob already exists" in message:
        message += " If you are sure this is new data, try renaming the file."
    response = jsonify(
        error=response_status_code,
        error_code=error_code,
        reason=reason,
        message=message,
    )
    response.status_code = status_code
    return response


def upload_to_blob_storage(yaml_data: yaml.YAMLObject, connection_string: str, blob_name: str):
    blob = BlobClient.from_connection_string(
        conn_str=connection_string, container_name="testfiles", blob_name=blob_name
    )
    # upload blob
    try:
        blob.upload_blob(json.dumps(yaml_data), overwrite=False)
        return http_response(200, "Success")
    except Exception as e:
        # TODO: specify the type of the exception more exactly, so we can be sure it has the fields assumed here.
        response = http_response(
            e.status_code,  # type: ignore
            e.message,  # type: ignore
            error_code=e.error_code,  # type: ignore
            response=e.response,  # type: ignore
            reason=e.reason,  # type: ignore
        )
        print(response)
        return response

    # List the blobs in the container
    # blob_list_after = container_client.list_blobs()


def insert_config_table_entry(connection_string: str, config_name: str, config_path: str):
    table_conn = TableService(connection_string=connection_string)
    new_entry = {
        "PartitionKey": "app",
        "RowKey": config_name,
        "Timestamp": datetime.now(),
        "ConfigName": config_name,
        "PathToBlob": config_path,
    }
    table_conn.insert_entity("abexconfigs", new_entry)


def insert_observation_table_entry(connection_string: str, file_name: str, file_path: str):
    table_conn = TableService(connection_string=connection_string)
    new_entry = {
        "PartitionKey": "app",
        "RowKey": file_name,
        "Timestamp": datetime.now(),
        "FileName": file_name,
        "PathToBlob": file_path,
    }
    table_conn.insert_entity("abexObservations", new_entry)


@app.route("/upload-config-data", methods=["GET", "POST"])
def upload_config_data():
    """
    Parse data into yaml and then upload to blob storage, as well as creating table entry
    """
    data = request.get_data()

    data_dict = parse_binary_to_dict(data)
    config_name: str = get_file_name(data_dict)  # type: ignore # auto
    config_data = get_config_data(data_dict)  # type: ignore # auto
    blob_name = config_name.split(".")[0]
    blob_path = "testfiles/" + blob_name

    # Upload the file to blob storage
    connection_string = get_connection_str_from_binary(data)
    upload_blob_response = upload_to_blob_storage(config_data, connection_string, blob_name)
    if upload_blob_response.status_code != 200:
        return upload_blob_response

    # TODO: move this once specs folders fixed
    # copy into abex specs folder
    new_spec_path = SPECS_DIR / config_name
    print("saving new spec to: ", new_spec_path)
    with open(new_spec_path, "w+") as f_path:
        yaml.dump(config_data, f_path)

    assert new_spec_path.is_file()

    # Add storage table entry
    insert_config_table_entry(connection_string, blob_name, blob_path)
    return {"filePath": config_name}


@app.route("/upload-observation-data", methods=["GET", "POST"])
def upload_observation_data():
    """
    Upload observations
    """
    data = request.get_data()
    data_dict = parse_binary_to_dict(data)
    print(f"data dict: {data_dict}")
    file_name = get_file_name(data_dict)  # type: ignore # auto
    csv_data: yaml.YAMLObject = get_csv_data(data_dict)  # type: ignore # auto

    blob_name = file_name.split(".")[0]
    blob_path = "testfiles/" + blob_name

    # Upload the file to blob storage
    connection_string = get_connection_str_from_binary(data)
    upload_blob_response = upload_to_blob_storage(csv_data, connection_string, blob_name)
    if upload_blob_response.status_code != 200:
        return upload_blob_response

    # Add storage table entry
    insert_observation_table_entry(connection_string, blob_name, blob_path)

    return {"filePath": file_name}


@app.route("/login/<string:connection_string>", methods=["GET"])
def login(connection_string: str):
    conn = from_connection_string(connection_string)
    if conn:
        print("conn successful")
    return {"success": True}


@app.route("/submit-new-experiment", methods=["GET", "POST"])
def submit_new_experiment():
    # TODO: start new experiment track
    """
    Submit a new experiment action.
    1. Retrieve the config from user's config table
    2. Retrieve the csv to user's csv table

    x. Create ABEX Config
    y. Submit the ABEX experiment
    """
    data = request.get_data()
    print(f"Data sent to submit-new-experiment: {data}")

    data_dict = parse_binary_to_dict(data)
    print(f"\ndata dict: {data_dict}")

    config_path = data_dict.get("configPath")  # type: ignore # auto
    # config_name = config_path.split('.')[0]
    print(f"config path: {config_path}")
    # observation_path = data_dict.get("observationsPath")

    yaml_file_path, config_dict = load_config_from_path_or_name(config_path)
    print(f"yaml file path: {yaml_file_path}")
    print(f"config dict: {config_dict}")

    for pair_list in load_resolutions(config_path):
        for _, config in pair_list:
            # Decide which optimization strategy should be used
            print(f"\nConfig: {config}")
            optimizer = OptimizerBase.from_strategy(config, config.optimization_strategy)
            optimizer.run()

    return data


@app.route("/submit-iteration", methods=["GET", "POST"])
def submit_iteration_form():
    # TODO: kick off new iteration
    data = request.get_data()
    print(data)
    return data


@app.route("/submit-clone", methods=["GET", "POST"])
def submit_cloneform():
    # TODO: kick off clone of previous experiment
    data = request.get_data()
    print(data)
    return data
