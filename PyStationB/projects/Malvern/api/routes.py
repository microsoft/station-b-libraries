# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import json
import sys
from werkzeug.datastructures import EnvironHeaders

from azure.cosmosdb.table.tableservice import TableService
from azure.storage.blob import BlobClient
from datetime import datetime
from flask import Response, jsonify, request
from pandas import DataFrame, read_csv
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from werkzeug.http import HTTP_STATUS_CODES
from werkzeug.wrappers.request import Request
from yaml import safe_load as yaml_safe_load
from yaml import dump as yaml_dump

from abex.simulations.run_simulator_pipeline import run_simulator_pipeline  # type: ignore  # noqa: E402
from abex.settings import load_config_from_path_or_name  # type: ignore  # noqa: E402  # auto
from psbutils.misc import ROOT_DIR  # type: ignore  # noqa: E402
from pyBCKG.azurestorage.api import from_connection_string  # type: ignore  # noqa: E402
from pyBCKG.utils import HttpRequestMethod  # type: ignore  # noqa: E402

from api import app  # type: ignore  # noqa: E402


MALVERN_DIR = ROOT_DIR / "projects" / "Malvern"
SIMULATION_SCRIPTS_DIR = ROOT_DIR / "projects" / "Malvern" / "simulator"
BLOB_CONTAINER = "testfiles"
CELLSIG_SCRIPTS_DIR = ROOT_DIR / "projects" / "CellSignaling" / "cellsignalling" / "cellsig_sim" / "scripts"
 
@app.route("/")  # type: ignore
@app.route("/get-experiment-options", methods=["GET"])  # type: ignore
def get_experiments() -> Response:
    """
    Retrieves a list of all stored experiments from the relevant Azure Table.

    Returns: Flask Response object containing the stored experiments

    """
    connection_string = request.headers.get("storageConnectionString")

    az_conn = from_connection_string(connection_string)
    query = "experiments()"
    # queryfilter = az_conn._queryfilter('deprecated', "False")
    # queryfilter = f"?$filter=Deprecated eq '{False}'"
    queryfilter = ""
    expt_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)
    experiments = expt_json["value"]
    # print(f"\nExpt options: {json.dumps(experiments)}")
    return Response(json.dumps(experiments), mimetype="application/json")


@app.route("/get-aml-runs", methods=["GET"])  # type: ignore
def get_aml_runs() -> Response:
    """
    Retrieves a list of Azure ML Runs (i.e. previously launched experiments) from the
    relevant Azure Table.

    Returns:
        Flask Response object containing the stored Azure ML RunIds and time of creation
    """
    connection_string = request.headers.get("storageConnectionString")

    az_conn = from_connection_string(connection_string)
    query = "azuremlruns()"
    queryfilter = ""
    aml_runs_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)

    aml_runs = aml_runs_json["value"]
    # print("\nAzureML Runs: ")
    # print(aml_runs)

    return Response(json.dumps(aml_runs), mimetype="application/json")


@app.route("/get-experiment-result", methods=["GET"])  # type: ignore
def get_experiment_results() -> Response:
    """
    Retrieves a list of stored Experiment Results from the relevant Azure Table. Differents from
    Experiments table as it also contains paths to results and suggested next experiments
    # TODO: This table only exists in dummy data - create in real.

    Returns:
        Flask Response containing the stored ExperimentResults.
    """

    connection_string = request.headers.get("storageConnectionString")
    experiment_name = request.args.get("experimentName")
    print(f"\n\nLooking for experiment name: {experiment_name}")

    az_conn = from_connection_string(connection_string)
    query = "experimentResults()"
    queryfilter = f"?$filter=RowKey%20eq%20'{experiment_name}'"

    experiment_results_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)
    experiment_results = experiment_results_json["value"]
    print(experiment_results)

    assert len(experiment_results) >0, "Error - no experiment exists with that rowKey"
    assert len(experiment_results) <2, "Error - more than 1 experiment exists with that RowKey"


    return Response(json.dumps(experiment_results), mimetype="application/json")


@app.route("/get-config-options", methods=["GET"])  # type: ignore
def get_configs() -> Response:
    """
    Retrieves a list of stored configs (name and path to location in Blob Storage) from the relevant Azure Table

    Returns:
        Flask Response containing the returned Config entries
    """
    # TODO: This table only exists in dummy data - create in real
    # configs = [{"id": "1", "name": "config1"}, {"id": "2", "name": "config2"}]
    connection_string = request.headers.get("storageConnectionString")
    az_conn = from_connection_string(connection_string)
    query = "abexconfigs()"
    queryfilter = ""

    experiment_results_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)
    configs = experiment_results_json["value"]
    # print("Configs: ", configs)
    return Response(json.dumps(configs), mimetype="application/json")


@app.route("/get-dataset-options", methods=["GET"])  # type: ignore
def get_datasets():
    """
    Retrieves a list of stored datasets (name and path to location in Blob Storage) from the relevant Azure Table

    Returns:
        Flask Response containing the returned Config entries
    """
    connection_string = request.headers.get("storageConnectionString")
    az_conn = from_connection_string(connection_string)
    query = "abexDatasets()"
    queryfilter = ""

    dataset_results_json = az_conn.query_table(HttpRequestMethod.GET, query, queryfilter)
    datasets = dataset_results_json["value"]
    return Response(json.dumps(datasets), mimetype="application/json")


def get_file_name(data_dict: Dict[str, Any]):
    """
    Given a dictionary retrieved from a Flask request, retrieve the value
    of the 'fileName' field

    Args:
        data_dict: the dictionary retrieved from the Flask request

    Returns:
        The config name
    """
    config = data_dict.get("config")
    assert config is not None
    config_data = json.loads(config)
    config_name = config_data.get("fileName")
    return config_name


def get_connection_str_from_request_headers(request: Request):
    """
    Get Azure Table Storage connection string from the headers of a Flask request

    Args:
        request: Flask request

    Returns:
        An Azure Storage connection string
    """
    headers: EnvironHeaders = request.headers
    connection_str = headers["storageConnectionString"]
    return connection_str


def http_response(status_code, message, error_code=None, reason=None):
    """
    Generate a Flask Response

    Args:
        status_code: HTTP status code of the request.
        message: message body
        error_code: Optional HTTP status related to the error
        reason: short description of the status code.

    Returns:
        [type]: [description]
    """
    response_status_code = HTTP_STATUS_CODES.get(status_code, "Unknown error")
    if status_code == 409 and "blob already exists" in message:
        message += " If you are sure this is new data, try renaming the file."
    response = jsonify(
        status_code=response_status_code,
        message=message,
        error_code=error_code,
        reason=reason,
    )
    response.status_code = status_code
    return response


def upload_to_blob_storage(data: str, connection_string: str, blob_name: str):
    """
    Upload data to blob storage

    Args:
        data: The data to upload
        connection_string: Connection string to connect to Azure Storage
        blob_name: Name to save the Blob as

    Returns:
        HTTP response demonstrating success or failure
    """
    blob = BlobClient.from_connection_string(
        conn_str=connection_string, container_name=BLOB_CONTAINER, blob_name=blob_name
    )
    # upload blob
    try:
        blob.upload_blob(data, overwrite=False)
        return http_response(200, "Success")
    except Exception as e:
        print(f"Exception: {e}")
        # TODO: specify the type of the exception more exactly, so we can be sure it has the fields assumed here.
        response = http_response(
            e.status_code,  # type: ignore
            e.message,  # type: ignore
            error_code=e.error_code,  # type: ignore
            reason=e.reason,  # type: ignore
        )
        print(response)
        return response

    # List the blobs in the container
    # blob_list_after = container_client.list_blobs()


def insert_config_table_entry(connection_string: str, config_name: str, config_path: str):
    """
    Insert entry to relevant Azure Table, recording timestamp, config name
    and path to config file in Blob Storage.

    Args:
        connection_string: A string containing credentials for connecting to Azure Table
        config_name: The name of the uploaded file containing the config
        config_path: The path in Blob Storage where the config is stored
    """
    table_conn = TableService(connection_string=connection_string)
    new_entry = {
        "PartitionKey": "app",
        "RowKey": config_name,
        "Timestamp": datetime.now(),
        "ConfigName": config_name,
        "PathToBlob": config_path,
    }
    table_conn.insert_entity("abexconfigs", new_entry)


def insert_dataset_table_entry(connection_string: str, file_name: str, file_path: str) -> None:
    """
    Insert entry to relevant dataset Azure Table, recording timestamp, observation file name
    and path to file to Blob Storage.

    Args:
        connection_string: A string containing credentials for connecting to Azure Table
        file_name: The name of the uploaded file containing observations
        file_path: The path in Blob Storage where the observations are stored
    """
    table_conn = TableService(connection_string=connection_string)
    new_entry = {
        "PartitionKey": "app",
        "RowKey": file_name,
        "Timestamp": datetime.now(),
        "FileName": file_name,
        "PathToBlob": file_path,
    }
    table_conn.insert_entity("abexDatasets", new_entry)


def insert_observation_table_entries(connection_string: str, entries_to_insert: DataFrame) -> None:
    """
    When user uploads CSV file of observations, add one entry per row into the relevant observations Azure Table.

    Args:
        connection_string: string containing credentials for connecting to Azure Table
        entries_to_insert: Pandas DataFrame of observations to be inserted into Azure Table
    """
    table_conn = TableService(connection_string=connection_string)
    print(f"entities: {entries_to_insert}")

    # TODO: insert entity must be either in dict format or entity object (cosmosdb table)
    records = entries_to_insert.to_dict(orient="records")

    def _insert_row(row):
        table_conn.insert_entity("abexObservations", row)  # pragma: no cover

    map(_insert_row, records)


@app.route("/upload-config-data", methods=["GET", "POST"])  # type: ignore
def upload_config_data() -> Dict[str, str]:
    """
    When user uploads a config file, insert into relevant Azure Table and store a copy locally,
    to use in AzureML.

    Returns:
        A dictionary containing the path to the config file locally.
    """
    # TODO: Return blobpath instead of filepath and confgure AML to read from Blob Storage
    config_data = request.get_data()
    print(f"Data: {config_data}")

    if request.method == "POST":
        print(f"Request: {request}")
        config, filename = save_and_read_uploaded_file(request)
        assert isinstance(filename, str)

        blob_name = filename.split(".")[0]
        blob_path = BLOB_CONTAINER + '/' + blob_name

        # Upload the file to blob storage
        connection_string = get_connection_str_from_request_headers(request)
        upload_blob_response = upload_to_blob_storage(
            json.dumps(config) + "This is a good request", connection_string, blob_name
        )
        if upload_blob_response.status_code != 200:
            return upload_blob_response  # type: ignore  # pragma: no cover

        # Add storage table entry
        insert_config_table_entry(connection_string, blob_name, blob_path)
        return {"filePath": filename}

    # otherwise GET request
    else:
        return {"filePath": ""}  # pragma: no cover


def update_config_folder(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update the config data to replace the 'folder' entry with the Malvern Uploads directory

    Args:
        config: The config read from the uploaded file

    Returns:
        The updated config
    """
    # TODO: Folder needs to be specific to experiment
    new_folder = Path(app.config["UPLOADS_DIR"])  # type: ignore
    new_folder.mkdir(exist_ok=True)
    config["data"]["folder"] = str(new_folder)
    return config


def save_and_read_uploaded_file(request: Request) -> Tuple[Union[DataFrame, Dict[str, Any]], str]:
    """
    Given a Flask HTTP request, retrieve the uploaded file (if existing), read it and save it in the uploads folder.
    Returns the data contained in the file (expected to be either a table of observations, or a config dictionary),
    plus the file name.

    Args:
        request: Flask HTTP request

    Returns:
        The contents of the uploaded file (either as a Pandas DataFrame, or a dictionary) and the filename
    """
    if request.files:
        Path(app.config["UPLOADS_DIR"]).mkdir(exist_ok=True)
        uploaded_file = request.files["file"]
        file_name: str = Path(uploaded_file.filename).name  # type: ignore
        filepath = Path(app.config["UPLOADS_DIR"]) / file_name
        uploaded_file.seek(0)  # type: ignore
        uploaded_file.save(filepath)
        if filepath.suffix == ".csv":
            with open(filepath, "r") as f_path:
                data = read_csv(f_path)
        elif filepath.suffix == ".yml":
            # Separately read and re-write the file
            with open(filepath, "r") as f_path:
                data = yaml_safe_load(f_path)
            # update the config to
            if "data" in data:
                if "folder" in data["data"]:  # pragma: no cover
                    data = update_config_folder(data)
                    with open(filepath, "w") as f_path:
                        yaml_dump(data, f_path)

        else:
            raise ValueError(f"Unrecognised file ending: {filepath.suffix}")
    else:
        raise ValueError("No file attached to request")
    uploaded_file_name = uploaded_file.filename
    assert isinstance(uploaded_file_name, str)
    return data, uploaded_file_name  # type: ignore


@app.route("/upload-observation-data", methods=["GET", "POST"])  # type: ignore
def upload_observation_data():
    """
    When user uploads a CSV file of observations, insert entry into relevant dataset Azure
    Table to track uploaded datasets, then parse the uploaded file and add rows as entries in
    the relevant observations Azure Table

    Returns:
        A dictionary containing the path to the observation file locally
    """
    data = request.get_data()
    # data_dict = json.loads(data)
    print(f"data dict: {data}")
    # file_name = get_file_name(data_dict)  # type: ignore # auto
    # print(f'CSV data: \n {csv_data}')

    if request.method == "POST":
        print(f"Request: {request}")
        observations, filename = save_and_read_uploaded_file(request)

        blob_name = filename.split(".")[0]
        # path in Azure Blob Storage
        # TODO: update this path
        blob_path = "testfiles/" + blob_name

        # Upload the file to blob storage
        connection_string = get_connection_str_from_request_headers(request)
        assert isinstance(observations, DataFrame)
        upload_blob_response = upload_to_blob_storage(json.dumps(observations.to_dict()), connection_string, blob_name)
        if upload_blob_response.status_code != 200:
            return upload_blob_response

        # Add storage table entry
        insert_dataset_table_entry(connection_string, blob_name, blob_path)

        # Add observations to table
        insert_observation_table_entries(connection_string, observations)

        return {"filePath": filename}

    # Otherwise GET request
    else:
        return {"filePath": ""}  # pragma: no cover


@app.route("/login/<string:connection_string>", methods=["GET"])  # type: ignore
def login(connection_string: str):
    """
    If connection string parses successfully, consider user to be logged in.
    #TODO: validate user permissions

    Args:
        connection_string: Azure Storage connection string

    Returns:
        success status (whether user was successfully logged in or not)
    """
    conn = from_connection_string(connection_string)
    if conn:
        print("conn successful")
        return {"success": True}
    else:
        return {"success": False}


@app.route("/submit-new-experiment", methods=["GET", "POST"])  # type: ignore
def submit_new_experiment():
    """
    Performs the following steps:
        1. Retrieve the config that the user uploaded/ selected
        2. Resolve all combinations of experiment arguments within the config
        3. Launch ABEX experiment (submits job to AML)

    Returns:
        Response containing details of the AML Run such as RunId and URL.
    """
    sys.path.append(str(ROOT_DIR / "projects"))
    from cellsignalling.cellsig_sim.optconfig import CellSignallingOptimizerConfig  # type: ignore  # noqa: E402
    data = request.get_data()
    print(f"Data sent to submit-new-experiment: {data}")

    if request.method == "POST":

        data_dict = json.loads(data)
        print(f"\ndata dict: {data_dict}")

        config_name = data_dict["headers"]["configPath"]  # type: ignore # auto
        config_path = Path(app.config["UPLOADS_DIR"]) / config_name
        print(f"config path: {config_path}")

        yaml_file_path, config_dict = load_config_from_path_or_name(config_path)
        print(f"yaml file path: {yaml_file_path}")
        print(f"config dict: {config_dict}")

        aml_config = data_dict["headers"]["amlConfig"]["data"]
        print(f"\nAML config: {aml_config}\n")

        arg_list = [
            "--spec_file",
            str(yaml_file_path),
            "--aml_root_dir",
            str(ROOT_DIR),
            "--num_iter",
            "2",
            "--num_runs",
            "2",
            "--plot_simulated_slices",
            "--submit_to_aml",
        ]

        azure_args = {
            "subscription_id": aml_config["SubscriptionId"],
            "resource_group": aml_config["ResourceGroup"],
            "workspace_name": aml_config["WorkspaceName"],
            "compute_target": aml_config["ComputeTarget"],
        }

        run_script_path =  CELLSIG_SCRIPTS_DIR / "run_cell_simulations_in_pipeline.py"
        plot_script_path =  CELLSIG_SCRIPTS_DIR / "plot_cellsig_predicted_optimum_convergence.py"
        loop_config_class = CellSignallingOptimizerConfig

        pipeline_run, pipeline = run_simulator_pipeline(
            arg_list, run_script_path, plot_script_path, loop_config_class, aml_arg_dict=azure_args
        )

        run_details = {
            "ExperimentName": pipeline_run.experiment.name,
            "RunId": pipeline_run._run_id,
            "RunUrl": pipeline_run._run_details_url,
        }

        print(f"Returning run details: \n{run_details}")

        return Response(json.dumps(run_details), mimetype="application/json")
    else:
        return {"filePath": None}  # pragma: no cover


@app.route("/submit-iteration", methods=["GET", "POST"])  # type: ignore
def submit_iteration_form():  # pragma: no cover
    # TODO: kick off new iteration
    data = request.get_data()
    print(data)
    return data


@app.route("/submit-clone/<string:aml_run_id>", methods=["GET", "POST"])  # type: ignore
def submit_clone_form(aml_run_id):  # pragma: no cover
    """
    Repeat a previously submitted Azure ML experiment Run.

    Args:
        aml_run_id: The Azure ML RunId of the Run to be repeated.

    Returns:
        [type]: [description]
    """
    # TODO: kick off clone of previous experiment
    data = request.get_data()
    print(data)
    return data


@app.route("/parse-aml-secrets", methods=["GET", "POST"])  # type: ignore
def parse_aml_secrets():
    """
    Upload user-defined file containing secrets necesary for submitting jobs to AML
    (including subscription id, resource group name, workspace name and compute target name)
    and store these secrets in

    Returns:
        HTTP response from submitting the request to upload
    """
    data = request.get_data()

    print(f"data dict: {data}")

    if request.method == "POST":
        aml_secrets, filename = save_and_read_uploaded_file(request)
        print(f"AML secrets: {aml_secrets}")
        aml_secrets = aml_secrets["variables"]
        subscription_id = aml_secrets["subscription_id"]
        resource_group = aml_secrets["resource_group"]
        workspace_name = aml_secrets["workspace_name"]
        compute_target = aml_secrets["compute_target"]
        # TODO: Store in props to be accessed when submitting experiment

        config = {
            "SubscriptionId": subscription_id,
            "ResourceGroup": resource_group,
            "WorkspaceName": workspace_name,
            "ComputeTarget": compute_target,
        }

        return Response(json.dumps(config), mimetype="application/json")

    # Otherwise GET request
    else:
        return {"success": False}  # pragma: no cover
