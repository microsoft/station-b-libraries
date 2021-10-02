# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import datetime
import json
import pytest
import sys

from io import BytesIO, StringIO
from typing import Any, Dict, Optional, Tuple

import yaml

from pandas import DataFrame
from pathlib import Path
from requests.models import Response
from unittest.mock import MagicMock, patch, mock_open

sys.path.append("C:\\Users\\mebristo\\PyStationB\\projects\\Malvern")
from api import app  # type: ignore  # noqa: E402
from api.routes import (  # type: ignore  # noqa: E402
    get_connection_str_from_request_headers,
    get_file_name,
    http_response,
    login,
    insert_config_table_entry,
    insert_dataset_table_entry,
    insert_observation_table_entries,
    save_and_read_uploaded_file,
    update_config_folder,
    upload_to_blob_storage,
)

CONN_STR = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=accountName.core.windows.net;"
    "AccountKey=accountKey.core.windowa.net;"
    "BlobEndpoint=blobEndpoint.core.windows;"
    "QueueEndpoint=queueEndpoint.core.windows.net;"
    "TableEndpoint=tableEndpoint.core.windows.net;"
    "FileEndpoint=fileEndpoint.core.windows.net;"
    "EndpointSuffix=core.windows.net;"
)


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as mock_client:
        yield mock_client


class MockAzureConnection:
    def __init__(self):
        pass

    def query_table(self, request_method, query, query_filter):
        if query == "experiments()":
            return {
                "value": [
                    {
                        "PartitionKey": "",
                        "RowKey": "storedExperiment1",
                        "Timestamp": str(datetime.datetime.now()),
                        "Name": "dummyExperiment6",
                        "Notes": "",
                        "Type": "Characterization",
                        "Deprecated": False,
                    }
                ]
            }
        elif query == "azuremlruns()":
            return {"value": [{"PartitionKey": "", "RowKey": "run1", "Timestamp": ""}]}
        elif query == "experimentResults()":
            expt_name = query_filter.replace("?$filter=RowKey%20eq%20'", "").replace("'", "")

            return {
                "value": [
                    {
                        "PartitionKey": "",
                        "RowKey": "experimentResult1",
                        "Timestamp": str(datetime.datetime.now()),
                        "Name": expt_name,
                        "Notes": "",
                        "Type": "Characterization",
                        "Deprecated": False,
                        "description": "",
                        "id": "",
                        # TODO: make this list
                        "folds": 1,
                        # TODO: make this list
                        "imageFolders": [],
                        # TODO: make this a list
                        "imageNames": [],
                        "iterations": 1,
                        "name": "",
                        "runId": "",
                        # TODO: make this list
                        "samples": [],
                        # TODO: make this list
                        "signals": [],
                        # TODO: suggestedExperiments should be list
                        "suggestedExperiments": [],
                        # TODO: What are abvailable image types?
                        "type": "",
                    }
                ]
            }
        elif query == "abexconfigs()":
            return {
                "value": [
                    {
                        "PartitionKey": "",
                        "RowKey": "config1",
                        "Timestamp": str(datetime.datetime.now()),
                        "Name": "dummyExperiment6",
                        "PathToBlob": "",
                    }
                ]
            }
        elif query == "abexDatasets()":
            return {
                "value": [
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
            }
        else:
            raise ValueError(f"Unrecognised query: {query}")

    def upload_blob(self, data: str, overwrite: Optional[bool] = False):

        if "This is a good request" in data:
            return {"success": True}
        else:
            raise MockException


class MockException(Exception):
    def __init__(self):
        self.status_code = 409
        self.error_code = 409
        self.message = "Blob already exists"
        self.reason = "Blob already exists"


class MockResponse(Response):
    def __init__(self, status_code=None, error_code=None, reason=None, message=None):
        self.status_code = status_code  # type: ignore
        self.error_code = error_code
        self.reason = reason  # type: ignore
        self.message = message


class MockBlobClient:
    def __init__(self):
        pass

    def from_connection_string(conn_str=None, container_name=None, blob_name=None):  # type: ignore
        return MockAzureConnection()


class MockTableService:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.mock_dataset_table = []
        self.mock_observation_table = []
        self.mock_config_table = []
        self.call_count = 0
        self.called = False

    def query_table_length(self, table_name):
        if table_name == "abexObservations":
            return len(self.mock_observation_table)
        elif table_name == "abexDatasets":
            return len(self.mock_dataset_table)
        elif table_name == "abexconfigs":
            return len(self.mock_config_table)
        else:
            raise ValueError(f"Cannot return length of table {table_name}")

    def insert_entity(self, table_name, entity: Tuple[Any]):
        if table_name == "abexObservations":
            table = self.mock_dataset_table
            self.mock_dataset_table
        elif table_name == "abexDatasets":
            table = self.mock_observation_table
        elif table_name == "abexconfigs":
            table = self.mock_config_table
        else:
            raise ValueError(f"Cannot insert entry into table {table_name}")
        table.append(entity)
        self.call_count += 1
        self.called = True


class MockPipeline:
    def __init__(self):
        pass

    def submit(self):
        pass


class MockExperiment:
    def __init__(self):
        self.name = "DummyExperiment"


class MockPipelineRun:
    def __init__(self):
        self.experiment = MockExperiment()
        self._run_id = "dummyId"
        self._run_details_url = "dummyUrl"


class MockCSVFile:
    def __init__(self):
        self.filename = "abc.csv"

    def seek(self, num):
        pass

    def save(self, path):
        pass


class MockYamlFile:
    def __init__(self):
        self.filename = "abc.yml"

    def seek(self, num):
        pass

    def save(self, path):
        pass


class MockTxtFile:
    def __init__(self):
        self.filename = "abc.txt"

    def seek(self, num):
        pass

    def save(self, path):
        pass


class MockRequest:
    def __init__(self):
        self.files = {"file": MockCSVFile()}

    def get_data(self):
        return {}


def mock_response():
    # if args.request_url == '/':
    response_content = json.dumps("a response")
    response = Response()
    response.status_code = 200
    response._content = str.encode(response_content)
    return response


def mock_connection(conn_str):
    return MockAzureConnection()


def test_get_experiments(client):
    """Start with a blank database."""
    req = MagicMock()

    req.headers.return_value = {"storageConnectionString": CONN_STR}

    with patch("api.routes.from_connection_string", mock_connection):
        with patch("flask.request", req):
            response = client.get("/get-experiment-options", headers={"storageConnectionString": CONN_STR})

    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert isinstance(response.json[0], dict)
    assert "RowKey" in response.json[0]
    assert response.json[0]["RowKey"] == "storedExperiment1"

    # without mock connection string, call to from_connection_string should fail
    with pytest.raises(Exception):
        client.get("/get-aml-runs")


def test_get_aml_runs(client):
    with patch("api.routes.from_connection_string", mock_connection):
        response = client.get("/get-aml-runs", headers={"storageConnectionString": CONN_STR})
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert isinstance(response.json[0], dict)
    assert "RowKey" in response.json[0]
    assert response.json[0]["RowKey"] == "run1"

    # without mock connection string, call to from_connection_string should fail
    with pytest.raises(Exception):
        client.get("/get-aml-runs")


def test_get_experiment_results(client):
    with patch("api.routes.from_connection_string", mock_connection):
        response = client.get(
            "/get-experiment-result",
            headers={"storageConnectionString": CONN_STR},
            query_string={"experimentName": "experiment1"},
        )
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert isinstance(response.json[0], dict)
    assert "RowKey" in response.json[0]
    assert response.json[0]["RowKey"] == "experimentResult1"


def test_get_configs(client):
    with patch("api.routes.from_connection_string", mock_connection):
        response = client.get("/get-config-options", headers={"storageConnectionString": CONN_STR})
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert isinstance(response.json[0], dict)
    assert "RowKey" in response.json[0]
    assert response.json[0]["RowKey"] == "config1"


def test_get_datasets(client):
    with patch("api.routes.from_connection_string", mock_connection):
        response = client.get("/get-dataset-options", headers={"storageConnectionString": CONN_STR})
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert isinstance(response.json[0], dict)
    assert len(response.json) == 2
    assert response.json[0]["name"] == "dataset1"


def test_get_file_name():
    data_dict = {"config": json.dumps({"fileName": "path123"})}
    assert get_file_name(data_dict) == "path123"


def test_get_connection_str_from_headers(client):

    req = MagicMock()

    headers = {"storageConnectionString": CONN_STR}
    req.headers.__getitem__.side_effect = headers.__getitem__

    conn_str = get_connection_str_from_request_headers(req)
    assert conn_str == CONN_STR

    with pytest.raises(Exception):
        client.get("/", headers={"storageConnectionString": "storageconnectionstring123"})


def mock_jsonify(status_code=None, message=None, error_code=None, reason=None):
    return MockResponse(status_code=status_code, error_code=error_code, reason=reason, message=message)


@patch("api.routes.jsonify", mock_jsonify)
def test_http_response():
    ok_response = http_response(200, "status is ok")
    assert ok_response.status_code == 200
    assert ok_response.message == "status is ok"

    bad_response = http_response(400, "bad request", error_code=400, reason="missing headers")
    assert bad_response.status_code == 400
    assert bad_response.message == "bad request"
    assert bad_response.error_code == 400
    assert bad_response.reason == "missing headers"

    bad_response = http_response(409, "Error - blob already exists", error_code=409)
    assert "If you are sure this is new data, try renaming the file." in bad_response.message
    assert bad_response.status_code == 409


def test_upload_to_blob_storage():
    connection_string = "connectionString123"
    blob_name = "blob_name123"
    with patch("api.routes.jsonify", mock_jsonify):
        with patch("api.routes.BlobClient", MockBlobClient):

            yaml_data_good_request = "This is a good request"
            response = upload_to_blob_storage(yaml_data_good_request, connection_string, blob_name)
            assert response.status_code == 200

            yaml_data_bad_request = "This is a bad request"
            response = upload_to_blob_storage(yaml_data_bad_request, connection_string, blob_name)
            assert response.error_code == 409

    with pytest.raises(Exception):
        upload_to_blob_storage({}, connection_string, blob_name)


def mock_insert(connection_string):
    return MockTableService(connection_string)


@patch("api.routes.TableService")
def test_insert_config_table_entry(mock_service):
    mock_service.side_effect = mock_insert
    connection_string = "connection123"
    config_name = "config1"
    config_path = Path("./tmpconfg")
    assert mock_service.call_count == 0
    _ = insert_config_table_entry(connection_string, config_name, config_path)
    assert mock_service.call_count == 1
    assert mock_service.called is True


@patch("api.routes.TableService")
def test_insert_dataset_table_entry(mock_service):
    mock_service.side_effect = mock_insert
    connection_string = "connection123"
    file_name = "file1"
    file_path = Path("./file1.csv")
    assert mock_service.call_count == 0
    # insert entry
    insert_dataset_table_entry(connection_string, file_name, file_path)
    assert mock_service.call_count == 1
    assert mock_service.called is True


@patch("api.routes.TableService")
def test_insert_observation_table_entries(mock_service):
    mock_service.side_effect = mock_insert
    connection_string = "connection123"
    table_to_insert = DataFrame(
        data=[(None, "row1", datetime.datetime.now(), None, None, 4.50, datetime.datetime.now(), False)],
        columns=["PartitionKey", "RowKey", "Timestamp", "sample", "signal", "value", "observedAt", "replicate"],
    )
    assert mock_service.call_count == 0
    insert_observation_table_entries(connection_string, table_to_insert)
    assert mock_service.call_count == 1
    assert mock_service.called is True


def get_mock_read_csv(path):
    return DataFrame({"a": [1, 2, 3]})


def get_mock_safeload(path):
    return {"a": [1, 2, 3]}


def test_update_config_folder():
    config_dict = {"data": {"folder": "somedir"}}
    updated_config = update_config_folder(config_dict)
    assert "uploads" in updated_config.get("data").get("folder")


@patch("api.routes.yaml_safe_load")
@patch("api.routes.read_csv")
@patch("api.routes.Path.mkdir")
def test_save_and_read_uploaded_file(mock_path_mkdir, mock_read_csv, mock_safeload):
    m = mock_open()
    mock_path_mkdir.side_effect = mock_mkdir
    mock_read_csv.side_effect = get_mock_read_csv
    mock_safeload.side_effect = get_mock_safeload
    # First test Mock request with CSV files as default
    mock_req = MockRequest()
    with patch("api.routes.open", m):
        with patch("api.routes.Path.exists", return_value=True):
            obs, filename = save_and_read_uploaded_file(mock_req)
            assert isinstance(obs, DataFrame)

    mock_req.files = {"file": MockYamlFile()}  # type: ignore
    with patch("api.routes.open", m):
        with patch("api.routes.Path.exists", return_value=True):
            config, filename = save_and_read_uploaded_file(mock_req)
            assert isinstance(config, Dict)

    mock_req.files = {"file": MockTxtFile()}  # type: ignore
    with patch("api.routes.open", m):
        with patch("api.routes.Path.exists", return_value=True):
            with pytest.raises(Exception):
                save_and_read_uploaded_file(mock_req)

    mock_req.files = None  # type: ignore
    with pytest.raises(Exception):
        save_and_read_uploaded_file(mock_req)


def mock_mkdir(exist_ok=True):
    return


def mock_save_and_upload_yaml_good_req(req):
    return {"a": [1, 2, "This is a good request"]}, "filename.yml"


@patch("api.routes.save_and_read_uploaded_file")
@patch("api.routes.Path.mkdir")
@patch("api.routes.TableService")
def test_upload_config_data(mock_service, mock_path_mkdir, mock_save_and_read, client):
    mock_save_and_read.side_effect = mock_save_and_upload_yaml_good_req
    mock_service.side_effect = mock_insert
    mock_path_mkdir.side_effect = mock_mkdir

    assert mock_service.call_count == 0

    dummy_filename = "config123"
    dummy_dict = {"data": {"folder": "somepath"}}

    str_buf = StringIO()
    # with open(dummy_filepath, 'w+') as f_path:
    yaml.dump(dummy_dict, str_buf)
    buf = BytesIO(str_buf.getvalue().encode("utf-8"))

    with patch("api.routes.BlobClient", MockBlobClient):
        with patch("api.routes.from_connection_string", mock_connection):
            response_good = client.post(
                "/upload-config-data",
                headers={"storageConnectionString": CONN_STR, "fileName": "config123.yml"},
                data={"file": (buf, f"{dummy_filename}.yml")},
            )

    assert response_good.status_code == 200
    assert isinstance(response_good.json, dict)
    # Check that inserted was called
    assert mock_service.call_count == 1
    assert mock_service.called is True


def mock_save_and_upload_bad_req(req):
    return DataFrame({"a": [1, 2, 3]}), "file.csv"


def mock_save_and_upload_good_req(req):
    return DataFrame({"a": [1, 2, "This is a good request"]}), "file.csv"


def get_mock_upload_blob(data, conn_str, blob_name):
    return MockResponse(status_code=200, message="success")


@patch("api.routes.save_and_read_uploaded_file")
@patch("api.routes.TableService")
def test_upload_observation_data(mock_service, mock_save_and_read, client, tmpdir):
    mock_service.side_effect = mock_insert

    # Create file of dummy observations
    dummy_filename = "file123"
    # dummy_file = FileStorage(filename=str(dummy_table_path), stream=buf, name=dummy_filename)
    with patch("api.routes.BlobClient", MockBlobClient):
        with patch("api.routes.from_connection_string", mock_connection):
            dummy_table_good = DataFrame(
                {"a": [1, 2, 3], "b": ["x", "y", "This is a good request"], "c": [45.0, 23.5, 234.7]}
            )
            str_buf = StringIO()
            dummy_table_good.to_csv(str_buf)
            buf_good = BytesIO(str_buf.getvalue().encode("utf-8"))
            # GOOD request
            mock_save_and_read.side_effect = mock_save_and_upload_good_req
            response_good = client.post(
                "/upload-observation-data",
                headers={"storageConnectionString": CONN_STR},
                data={"file": (buf_good, f"{dummy_filename}.csv")},
            )
            assert response_good.status_code == 200
            assert isinstance(response_good.json, dict)

            dummy_table_bad = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [45.0, 23.5, 234.7]})
            str_buf = StringIO()
            dummy_table_bad.to_csv(str_buf)
            buf_bad = BytesIO(str_buf.getvalue().encode("utf-8"))
            # GOOD request
            mock_save_and_read.side_effect = mock_save_and_upload_bad_req
            response_bad = client.post(
                "/upload-observation-data",
                headers={"storageConnectionString": CONN_STR},
                data={"file": (buf_bad, f"{dummy_filename}.csv")},
            )
            assert response_bad.status_code == 409


def test_login(client):
    with patch("api.routes.BlobClient", MockBlobClient):
        with patch("api.routes.from_connection_string", mock_connection):
            response = client.get(
                f"/login/{CONN_STR}",
            )
            assert isinstance(response.json, dict)
            assert response.json.get("success") is True

        with patch("api.routes.from_connection_string", return_value=None):
            bad_response = client.get(
                f"/login/{CONN_STR}",
            )
            assert isinstance(bad_response.json, dict)
            assert bad_response.json.get("success") is False

    with pytest.raises(Exception):
        login("abc")


def mock_simulator_pipeline(arg_list, run_script_path, plot_script_path, loop_config_class, aml_arg_dict):
    return MockPipelineRun(), MockPipeline()


@patch("api.routes.run_simulator_pipeline")
def test_submit_new_experiment(mock_sim_pipeline, client):
    def _mock_config_path_and_dict(conf_path):
        return conf_path, {"data"}

    mock_sim_pipeline.side_effect = mock_simulator_pipeline

    with patch("api.routes.load_config_from_path_or_name", _mock_config_path_and_dict):
        with patch("abex.simulations.run_simulator_pipeline.Pipeline", MockPipeline):
            with patch("api.routes.BlobClient", MockBlobClient):
                with patch("api.routes.from_connection_string", mock_connection):
                    new_experiment = client.post(
                        "/submit-new-experiment",
                        json={
                            "headers": {
                                "storageConnectionString": CONN_STR,
                                "configPath": "conf123",
                                "observationsPath": "obs123",
                                "amlConfig": {
                                    "data": {
                                        "SubscriptionId": "sub",
                                        "ResourceGroup": "rg",
                                        "WorkspaceName": "wn",
                                        "ComputeTarget": "ct",
                                    }
                                },
                            }
                        },
                    )
    assert new_experiment.status_code == 200
    assert isinstance(new_experiment.json, dict)


def mock_save_and_upload_aml(req):
    return {
        "variables": {"subscription_id": "sub", "resource_group": "rg", "workspace_name": "ws", "compute_target": "ct"}
    }, "az.yml"


@patch("api.routes.save_and_read_uploaded_file")
def test_parse_aml_secrets(mock_save_and_read, client):
    mock_save_and_read.side_effect = mock_save_and_upload_aml
    dummy_dict = {
        "variables": {"subscription_id": "sub", "resource_group": "rg", "workspace_name": "ws", "compute_target": "ct"}
    }
    dummy_filename = "abc"
    str_buf = StringIO()
    yaml.dump(dummy_dict, str_buf)
    buf = BytesIO(str_buf.getvalue().encode("utf-8"))

    with patch("api.routes.BlobClient", MockBlobClient):
        with patch("api.routes.from_connection_string", mock_connection):
            response_good = client.post(
                "/parse-aml-secrets",
                headers={"storageConnectionString": CONN_STR},
                data={"file": (buf, f"{dummy_filename}.yml")},
            )

    assert response_good.status_code == 200
