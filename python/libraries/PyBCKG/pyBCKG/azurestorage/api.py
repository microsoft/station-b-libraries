# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
Exports:
    AzureConnection, connection object used to retrieve the data from BCKG
    from_connection_string, a factory method creating AzureConnection objects
"""
import base64
import hashlib
import hmac
import json
from ast import literal_eval
from io import StringIO
from typing import Dict, Iterable, List, Optional, Set, Tuple
from itertools import groupby

import pyBCKG.domain as domain
import pyBCKG.memory
import requests
from pyBCKG.utils import HttpRequestMethod, get_rfc_date_time

X_MS_VERSION = "2019-07-07"
PROCESS_DATA_EVENT = "ProcessDataEvent"
SAMPLE_DATA_EVENT = "SampleDataEvent"
NEXT_PARTITION_KEY = "x-ms-continuation-NextPartitionKey"
NEXT_ROW_KEY = "x-ms-continuation-NextRowKey"


class AzureConnection:
    def __init__(
        self,
        default_endpoints_protocol: str,
        account_name: str,
        account_key: str,
        blob_endpoint: str,
        queue_endpoint: str,
        table_endpoint: str,
        file_endpoint: str,
        verbose: bool = True,
    ) -> None:
        self.default_endpoints_protocol: str = default_endpoints_protocol
        self.account_name: str = account_name
        self.account_key: str = account_key
        self.blob_endpoint: str = blob_endpoint
        self.queue_endpoint: str = queue_endpoint
        self.table_endpoint: str = table_endpoint
        self.file_endpoint: str = file_endpoint
        self.verbose: bool = verbose

        self.localstorage = pyBCKG.memory.MemoryStorage()

    def get_blob(self, query: str) -> StringIO:
        request = HttpRequestMethod.GET
        now = get_rfc_date_time()
        canonicalized_headers = f"x-ms-date:{now}\nx-ms-version:{X_MS_VERSION}\n"
        string_to_sign = f"{request.value}\n\n\n\n{canonicalized_headers}/{self.account_name}/{query}"
        message = bytes(string_to_sign, "utf-8")
        secret = base64.b64decode(self.account_key)

        signature = base64.b64encode(hmac.new(secret, message, digestmod=hashlib.sha256).digest())
        apikey = f"SharedKeyLite {self.account_name}:{signature.decode('utf-8')}"

        url = self.blob_endpoint.strip() + query
        headers = {
            "Authorization": apikey,
            "x-ms-date": now,
            "x-ms-version": X_MS_VERSION,
        }
        r = requests.get(url, headers=headers)
        return StringIO(r.text)

    def get_timeseries_file(self, file_id: str) -> StringIO:
        endpoint = f"timeseries/{file_id}"
        return self.get_blob(endpoint)

    # Wrappers for BCKG Specific Function calls.
    # GET METHODS
    @staticmethod
    def _queryfilter(column: str, value: str) -> str:
        return f"?$filter={column} eq '{value}'"

    @staticmethod
    def _querydisjunction(queries: List[Tuple[str, str]]) -> str:
        filter = " or ".join([f"{x} eq '{y}'" for x, y in queries])
        return f"?$filter={filter}"

    @staticmethod
    def _queryconjunction(queries: List[Tuple[str, str]]) -> str:
        filter = " and ".join([f"{x} eq '{y}'" for x, y in queries])
        return f"?$filter={filter}"

    @staticmethod
    def reagent_from_properties(xres: dict) -> domain.Reagent:
        guid = xres["RowKey"]
        name = xres["Name"]
        notes = xres["Notes"]
        barcode = xres["Barcode"]

        reagent: domain.Reagent
        if xres["Type"] == "Chemical":
            chem_type = domain.ChemicalType(xres["ChemicalType"])
            reagent = domain.Chemical(guid=guid, name=name, notes=notes, _type=chem_type, barcode=barcode)
            return reagent
        elif xres["Type"] == "DNA":
            dna = xres["Sequence"]
            dna_type = domain.DNAType(xres["DNAType"])

            reagent = domain.DNA(
                guid=guid,
                name=name,
                notes=notes,
                dna=dna,
                _type=dna_type,
                barcode=barcode,
            )
            return reagent
        raise NotImplementedError(f"{xres['Type']} type of Reagent not implemented yet.")

    @staticmethod
    def condition_from_properties(xres: dict, reagent_map: Dict[str, domain.Reagent]) -> domain.Condition:
        """Utility function which parses a dictionary into a domain.Condition."""
        guid = xres["RowKey"]
        concentration = domain.Concentration(value=xres["value"], units=xres["valueUnits"])
        reagent = reagent_map[xres["reagentId"]]

        if xres["time"] == -1.0:
            time_value = None
        else:
            time_value = domain.Time(value=xres["time"], units=xres["timeUnits"])
        return domain.Condition(guid=guid, reagent=reagent, concentration=concentration, time=time_value)

    def query_table(self, request, query, query_filter):
        dataformat = "application/json;odata=nometadata"
        now = get_rfc_date_time()
        string_to_sign = f"{request.value}\n\n{dataformat}\n{now}\n/{self.account_name}/{query}"
        message = bytes(string_to_sign, "utf-8")
        secret = base64.b64decode(self.account_key)

        signature = base64.b64encode(hmac.new(secret, message, digestmod=hashlib.sha256).digest())
        apikey = f"SharedKey {self.account_name}:{signature.decode('utf-8')}"

        url = self.table_endpoint.strip() + query + query_filter
        headers = {
            "Authorization": apikey,
            "Accept": dataformat,
            "x-ms-date": now,
            "Content-Type": dataformat,
            "x-ms-version": X_MS_VERSION,
            "DataServiceVersion": "3.0;NetFx",
            "MaxDataServiceVersion": "3.0;NetFx",
        }
        r = requests.get(url, headers=headers)
        jsonval = json.loads(r.text)
        result_list = jsonval["value"]
        while NEXT_PARTITION_KEY in r.headers and NEXT_ROW_KEY in r.headers:
            next_partition_key = r.headers[NEXT_PARTITION_KEY]
            next_row_key = r.headers[NEXT_ROW_KEY]
            next_filter = f"?NextPartitionKey={next_partition_key}&NextRowKey={next_row_key}"
            next_url = self.table_endpoint.strip() + query + next_filter
            r = requests.get(next_url, headers=headers)
            next_jsonval = json.loads(r.text)
            result_list += next_jsonval["value"]
        final_result = {"value": result_list}
        return final_result

    def get_reagent_by_id(self, reagent_id: str) -> domain.Reagent:
        if reagent_id in self.localstorage.reagents:
            return self.localstorage.reagents[reagent_id]

        query = "reagents()"
        queryfilter = self._queryfilter("RowKey", reagent_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        xres = x["value"][0]
        reagent = AzureConnection.reagent_from_properties(xres)
        self.localstorage.add_reagent(reagent)
        return reagent

    def get_reagents(self, reagent_ids: Set[str]) -> List[domain.Reagent]:
        existing = [self.localstorage.reagents[r_id] for r_id in reagent_ids if r_id in self.localstorage.reagents]
        new_reagent_query = [("RowKey", r_id) for r_id in reagent_ids if r_id not in self.localstorage.reagents]
        if len(new_reagent_query) > 1:
            query = "reagents()"
            queryfilter = self._querydisjunction(new_reagent_query)
            x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
            new_reagents = [AzureConnection.reagent_from_properties(xres) for xres in x["value"]]
            for reagent in new_reagents:
                self.localstorage.add_reagent(reagent)
            return existing + new_reagents
        else:
            return existing

    def _parse_condition(self, xres: dict) -> domain.Condition:
        """Utility function which parses a dictionary into a domain.Condition."""
        guid = xres["RowKey"]
        concentration = domain.Concentration(value=xres["value"], units=xres["valueUnits"])
        reagent = self.get_reagent_by_id(xres["reagentId"])

        if xres["time"] == -1.0:
            time_value = None
        else:
            time_value = domain.Time(value=xres["time"], units=xres["timeUnits"])
        return domain.Condition(guid=guid, reagent=reagent, concentration=concentration, time=time_value)

    def get_condition_by_id(self, condition_id: str):
        query = "sampleconditions()"
        queryfilter = self._queryfilter("RowKey", condition_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        xres = x["value"][0]
        return self._parse_condition(xres)

    @staticmethod
    def cell_entity_from_properties(xres: dict, reagent_map: Dict[str, domain.Reagent]) -> domain.CellEntity:
        compartment = domain.Compartment(xres["compartment"])
        entity = reagent_map[xres["entity"]]
        return domain.CellEntity(compartment=compartment, entity=entity)

    def get_cell_entities(self, cell_ids: Set[str]) -> Dict[str, List[domain.CellEntity]]:
        query = "cellentities()"
        queryfilter = self._querydisjunction([("cellId", cell_id) for cell_id in cell_ids])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        reagent_ids = {xres["entity"] for xres in x["value"]}
        reagent_map = {r.guid: r for r in self.get_reagents(reagent_ids)}
        cell_groups = groupby(
            sorted(x["value"], key=lambda k: k["cellId"]),
            lambda x: x["cellId"],
        )
        all_cell_entities: Dict[str, List[domain.CellEntity]] = {
            c_id: [AzureConnection.cell_entity_from_properties(props, reagent_map) for props in xres]
            for c_id, xres in cell_groups
        }
        return all_cell_entities

    @staticmethod
    def cell_from_properties(xres, cell_entities_map: Dict[str, List[domain.CellEntity]]) -> domain.Cell:
        guid: str = xres["RowKey"]
        name = xres["Name"]
        notes = xres["Notes"]
        entities = cell_entities_map.get(guid, [])
        return domain.Cell(guid=guid, name=name, notes=notes, entities=entities)

    def get_cell_by_id(self, cell_id: str) -> domain.Cell:
        if cell_id in self.localstorage.cells:
            return self.localstorage.cells[cell_id]

        query = "cells()"
        queryfilter = self._queryfilter("RowKey", cell_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        cell_entities_map = self.get_cell_entities({cell_id})
        cell = AzureConnection.cell_from_properties(x["value"], cell_entities_map)
        self.localstorage.add_cell(cell)
        return cell

    def get_cells(self, cell_ids: Set[str]) -> List[domain.Cell]:
        existing = [self.localstorage.cells[c_id] for c_id in cell_ids if c_id in self.localstorage.cells]
        new_cell_ids = {c_id for c_id in cell_ids if c_id not in self.localstorage.cells}
        new_cells_query = [("RowKey", c_id) for c_id in new_cell_ids]
        if len(new_cells_query) > 0:
            query = "cells()"
            queryfilter = self._querydisjunction(new_cells_query)
            x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
            cell_entities_map = self.get_cell_entities(new_cell_ids)
            new_cells = [AzureConnection.cell_from_properties(xres, cell_entities_map) for xres in x["value"]]
            for cell in new_cells:
                self.localstorage.add_cell(cell)
            return existing + new_cells
        else:
            return existing

    @staticmethod
    def device_from_properties(xres, cell_map: Dict[str, domain.Cell]) -> domain.SampleDevice:
        cell = cell_map[xres["cellId"]]
        cell_density = xres.get("cellDensity", None)
        preseeding_density = xres.get("cellPreSeeding", None)
        if preseeding_density == -1:
            preseeding_density = None
        return domain.SampleDevice(
            cell=cell,
            cell_density=cell_density,
            preseeding_density=preseeding_density,
        )

    def get_sample_device(self, sample_id: str) -> Optional[domain.SampleDevice]:  # type: ignore
        query = "sampledevices()"
        queryfilter = self._queryfilter("sampleId", sample_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        if len(x["value"]) == 0:
            return None
        else:
            xres = x["value"][0]
            cell = self.get_cell_by_id(xres["cellId"])
            return AzureConnection.device_from_properties(xres, {cell.guid: cell})

    def get_all_devices_by_sample_ids(self, sample_ids: Set[str]) -> Dict[str, domain.SampleDevice]:
        query = "sampledevices()"
        queryfilter = AzureConnection._querydisjunction([("sampleId", sample_id) for sample_id in sample_ids])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        cell_ids = {xres["cellId"] for xres in x["value"]}
        cell_map = {c.guid: c for c in self.get_cells(cell_ids)}
        sample_groups = groupby(
            sorted(x["value"], key=lambda k: k["sampleId"]),
            lambda x: x["sampleId"],
        )
        all_devices: Dict[str, domain.SampleDevice] = {
            s_id: AzureConnection.device_from_properties(list(xres)[0], cell_map) for s_id, xres in sample_groups
        }
        return all_devices

    @staticmethod
    def filter_from_properties(filter_val: str) -> domain.PlateReaderFilter:
        split = filter_val.split("-")
        midpoint = float(split[0])
        width = float(split[1]) if len(split) > 1 else None
        return domain.PlateReaderFilter(midpoint=midpoint, width=width)

    @staticmethod
    def signal_from_properties(xres: dict) -> domain.Signal:
        guid = xres["id"]
        signal_type = xres["Type"]  # TODO Add Azure columns names as Enums

        if signal_type == "Titre":  # TODO Add Signal Types as Enums
            return domain.Titre(guid=guid)
        elif signal_type == "CellDiameter":
            return domain.CellDiameter(guid=guid)
        elif signal_type == "Aggregation":
            return domain.Aggregation(guid=guid)
        elif signal_type == "PlateReaderFluorescence":
            emission = AzureConnection.filter_from_properties(xres["emission"])
            excitation = AzureConnection.filter_from_properties(xres["excitation"])
            gain = float(xres["gain"]) if xres.get("gain") != -1 else None
            return domain.PlateReaderFluorescence(guid=guid, emission=emission, excitation=excitation, gain=gain)
        elif signal_type == "PlateReaderAbsorbance":
            wavelength = float(xres["wavelength"])
            correction = float(xres["correction"]) if xres.get("correction") != -1 else None
            gain = float(xres["gain"]) if xres.get("gain") != -1 else None
            return domain.PlateReaderAbsorbance(guid=guid, wavelength=wavelength, correction=correction, gain=gain)
        elif signal_type == "PlateReaderLuminescence":
            return domain.PlateReaderLuminescence(guid=guid)
        elif signal_type == "PlateReaderTemperature":
            return domain.PlateReaderTemperature(guid=guid)
        elif signal_type.startswith("Generic"):
            signal_name = signal_type.split(":")[1]
            return domain.GenericSignal(guid=guid, name=signal_name)
        elif signal_type.startswith("CellCount"):
            cell_count_type = domain.CellCountType(signal_type.split(":")[1])
            return domain.CellCount(guid=guid, cellCountType=cell_count_type)
        elif signal_type.startswith("TransfectionEfficiency"):
            gene = signal_type.split(":")[1]
            return domain.TransfectionEfficiency(guid=guid, gene=gene)
        else:
            raise ValueError(f"{signal_type} not a recognized Signal")

    def get_signal_by_rowkey(self, signal_id: str) -> domain.Signal:
        query = "signals()"
        queryfilter = self._queryfilter("RowKey", signal_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        xres = x["value"][0]
        signal = AzureConnection.signal_from_properties(xres)

        # Add the signal to the local storage
        self.localstorage.add_signal(signal)
        return signal

    def get_signal_by_id(self, signal_id: str) -> domain.Signal:
        query = "signals()"
        queryfilter = self._queryfilter("id", signal_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        xres = x["value"][0]
        signal = AzureConnection.signal_from_properties(xres)
        self.localstorage.add_signal(signal)
        return signal

    def get_signals(self, signal_ids: Set[str]) -> List[domain.Signal]:
        existing = [self.localstorage.signals[s_id] for s_id in signal_ids if s_id in self.localstorage.signals]
        new_signal_ids = {s_id for s_id in signal_ids if s_id not in self.localstorage.signals}
        new_signal_filters = [("id", s_id) for s_id in new_signal_ids]
        if len(new_signal_filters) > 0:
            query = "signals()"
            queryfilter = AzureConnection._querydisjunction(new_signal_filters)
            x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
            new_signals = [AzureConnection.signal_from_properties(xres) for xres in x["value"]]
            for signal in new_signals:
                self.localstorage.add_signal(signal)
            return existing + new_signals
        else:
            return existing

    def get_sample_conditions(self, sample_id: str) -> List[domain.Condition]:
        query = "sampleConditions()"
        queryfilter = self._queryfilter("sampleId", sample_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)

        conditions = [self._parse_condition(xres) for xres in x["value"]]
        return conditions

    def get_all_conditions_by_sample_ids(self, sample_ids: Set[str]) -> Dict[str, List[domain.Condition]]:
        query = "sampleConditions()"
        queryfilter = AzureConnection._querydisjunction([("sampleId", sample_id) for sample_id in sample_ids])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        reagent_ids = {xres["reagentId"] for xres in x["value"]}
        reagent_map = {r.guid: r for r in self.get_reagents(reagent_ids)}
        sample_groups = groupby(
            sorted(x["value"], key=lambda k: k["sampleId"]),
            lambda x: x["sampleId"],
        )
        all_conditions: Dict[str, List[domain.Condition]] = {
            s_id: [AzureConnection.condition_from_properties(props, reagent_map) for props in xres]
            for s_id, xres in sample_groups
        }
        return all_conditions

    @staticmethod
    def observation_from_properties(xres: dict, signal_map: Dict[str, domain.Signal]) -> domain.Observation:
        signal = signal_map[xres["signal"]]
        observed_at = xres.get("observedAt", None) or None
        replicate = xres.get("replicate", None) or None
        units = xres.get("units", None) or None
        measure = xres.get("measure", None) or None
        if measure is not None:
            measure = domain.MeasureType(measure)
        measuredBy = xres.get("measuredBy", None) or None
        observation = domain.Observation(
            xres["RowKey"],
            xres["value"],
            replicate,  # type: ignore # auto
            observed_at,  # type: ignore # auto
            signal,
            units,
            measure,
            measuredBy,
        )
        return observation

    def get_sample_observations(self, sample_id: str) -> List[domain.Observation]:
        query = "observations()"
        queryfilter = self._queryfilter("sample", sample_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        observations = []
        signal_ids = {xres["signal"] for xres in x["value"]}
        signal_map = {s.guid: s for s in self.get_signals(signal_ids)}
        observations = [AzureConnection.observation_from_properties(xres, signal_map) for xres in x["value"]]
        return observations

    def get_all_observations_by_sample_ids(self, sample_ids: Set[str]) -> Dict[str, List[domain.Observation]]:
        query = "observations()"
        queryfilter = AzureConnection._querydisjunction([("sample", sample_id) for sample_id in sample_ids])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        signal_ids = {xres["signal"] for xres in x["value"]}
        signal_map = {s.guid: s for s in self.get_signals(signal_ids)}

        sample_groups = groupby(
            sorted(x["value"], key=lambda k: k["sample"]),
            lambda x: x["sample"],
        )
        all_observations: Dict[str, List[domain.Observation]] = {
            s_id: [AzureConnection.observation_from_properties(props, signal_map) for props in xres]
            for s_id, xres in sample_groups
        }
        return all_observations

    def list_file_ids_of(self, guid: str) -> List[str]:
        query = "filesmap()"
        queryfilter = self._queryfilter("source", guid)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        file_ids = [x["RowKey"] for x in x["value"]]
        return file_ids

    def get_latest_process_data_event(self, experiment_id: str) -> str:
        query = "eventsV2()"
        queryfilter = self._queryconjunction([("targetId", experiment_id), ("targetType", PROCESS_DATA_EVENT)])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        return max(x["value"], key=lambda item: item["timestamp"])["RowKey"]

    def get_data_list_triggered_by(self, triggered_by: str) -> Dict[str, str]:
        """Returns a dictionary of sample-timeseries files which were triggered by a specific Process Data Event

        Args:
            triggered_by: ID of the Process Data Event.

        Returns:
            dictionary of sample data where the key is the Sample Id and the value is the associated timeseries File Id
        """
        query = "eventsV2()"
        queryfilter = self._queryconjunction([("triggeredBy", triggered_by), ("targetType", SAMPLE_DATA_EVENT)])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        sample_data = {}
        for xres in x["value"]:
            sampleId = xres["targetId"]
            file_id = {k: v for (k, v) in literal_eval(xres["change"])}["++fileId"]
            sample_data[sampleId] = file_id
        return sample_data

    def get_experiment_signals(self, experiment_id: str) -> List[domain.Signal]:
        query = "signals()"
        queryfilter = self._queryfilter("experimentId", experiment_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        signals = [AzureConnection.signal_from_properties(xres) for xres in x["value"]]
        return signals

    def get_experiment_signals_by_expt_ids(self, experiment_ids: Set[str]) -> Dict[str, List[domain.Signal]]:
        if len(experiment_ids) == 0:
            return {}
        query = "signals()"
        signal_filters = [("experimentId", expt_id) for expt_id in experiment_ids]
        queryfilter = self._querydisjunction(signal_filters)
        res = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        expt_groups = groupby(
            sorted(res["value"], key=lambda k: k["experimentId"]),
            lambda x: x["experimentId"],
        )
        all_expt_signals: Dict[str, List[domain.Signal]] = {
            e_id: [AzureConnection.signal_from_properties(props) for props in xres] for e_id, xres in expt_groups
        }
        return all_expt_signals

    @staticmethod
    def experiment_operation_from_properties(xres) -> domain.ExperimentOperation:
        guid = xres["RowKey"]
        _type = domain.ExperimentOperationType(xres["Type"])
        timestamp = xres["triggerTime"]
        experiment_operation = domain.ExperimentOperation(guid=guid, _type=_type, timestamp=timestamp)
        return experiment_operation

    def get_experiment_operations(self, experiment_id: str) -> List[domain.ExperimentOperation]:
        query = "experimentevents()"
        queryfilter = self._queryfilter("source", experiment_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        operations = [AzureConnection.experiment_operation_from_properties(xres) for xres in x["value"]]
        return operations

    def get_experiment_operations_by_expt_ids(
        self, experiment_ids: Set[str]
    ) -> Dict[str, List[domain.ExperimentOperation]]:
        if len(experiment_ids) == 0:
            return {}
        query = "experimentevents()"
        expt_op_filters = [("source", expt_id) for expt_id in experiment_ids]
        queryfilter = self._querydisjunction(expt_op_filters)
        res = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        expt_groups = groupby(
            sorted(res["value"], key=lambda k: k["source"]),
            lambda x: x["source"],
        )
        all_expt_ops: Dict[str, List[domain.ExperimentOperation]] = {
            e_id: [AzureConnection.experiment_operation_from_properties(props) for props in xres]
            for e_id, xres in expt_groups
        }
        return all_expt_ops

    @staticmethod
    def sample_from_properties(
        xres: dict,
        condition_map: Dict[str, List[domain.Condition]],
        observation_map: Dict[str, List[domain.Observation]],
        device_map: Dict[str, domain.SampleDevice],
    ) -> domain.Sample:
        guid = xres["RowKey"]
        physical_well = domain.Position(row=xres["physicalWellRow"], col=xres["physicalWellCol"])
        virtual_well = domain.Position(row=xres["virtualWellRow"], col=xres["virtualWellCol"])
        conditions = condition_map.get(guid, [])
        observations = observation_map.get(guid, [])
        device = device_map[guid]
        sample = domain.Sample(
            guid=guid,
            physical_plate_name=xres["physicalPlateName"],
            physical_well=physical_well,
            virtual_well=virtual_well,
            device=device,
            conditions=conditions,
            observations=observations,
        )
        return sample

    def get_sample_by_id(self, sample_id: str) -> domain.Sample:
        query = "samples()"
        queryfilter = self._queryfilter("RowKey", sample_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        xres = x["value"][0]
        guid = xres["RowKey"]
        sample_condition_map = self.get_all_conditions_by_sample_ids({guid})
        sample_observation_map = self.get_all_observations_by_sample_ids({guid})
        sample_device_map = self.get_all_devices_by_sample_ids({guid})
        return AzureConnection.sample_from_properties(
            xres, sample_condition_map, sample_observation_map, sample_device_map
        )

    def get_experiment_samples(self, experiment_id: str) -> List[domain.Sample]:
        query = "samples()"
        queryfilter = self._queryfilter("experimentId", experiment_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)

        sample_ids = {jsonval["RowKey"] for jsonval in x["value"]}
        sample_condition_map = self.get_all_conditions_by_sample_ids(sample_ids)
        sample_observation_map = self.get_all_observations_by_sample_ids(sample_ids)
        sample_device_map = self.get_all_devices_by_sample_ids(sample_ids)
        samples = [
            AzureConnection.sample_from_properties(
                xres, sample_condition_map, sample_observation_map, sample_device_map
            )
            for xres in x["value"]
        ]
        return samples

    def get_experiment_samples_by_expt_ids(self, experiment_ids: Set[str]) -> Dict[str, List[domain.Sample]]:
        if len(experiment_ids) == 0:
            return {}
        query = "samples()"
        sample_query = [("experimentId", expt_id) for expt_id in experiment_ids]
        queryfilter = self._querydisjunction(sample_query)
        res = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        sample_ids: List[str] = [jsonval["RowKey"] for jsonval in res["value"]]
        n = 300  # Should experiment to see if we can stretch this to maybe 300?
        split_samples = [
            set(sample_ids[i * n : (i + 1) * n]) for i in range((len(sample_ids) + n - 1) // n)  # noqa: E203
        ]
        sample_condition_map: Dict[str, List[domain.Condition]] = {}
        sample_observation_map: Dict[str, List[domain.Observation]] = {}
        sample_device_map: Dict[str, domain.SampleDevice] = {}
        for sample_id_set in split_samples:
            sample_condition_map.update(self.get_all_conditions_by_sample_ids(sample_id_set))
            sample_observation_map.update(self.get_all_observations_by_sample_ids(sample_id_set))
            sample_device_map.update(self.get_all_devices_by_sample_ids(sample_id_set))
        expt_groups = groupby(
            sorted(res["value"], key=lambda k: k["experimentId"]),
            lambda x: x["experimentId"],
        )
        all_expt_samples: Dict[str, List[domain.Sample]] = {
            e_id: [
                AzureConnection.sample_from_properties(
                    props, sample_condition_map, sample_observation_map, sample_device_map
                )
                for props in xres
            ]
            for e_id, xres in expt_groups
        }
        return all_expt_samples

    def list_entity_ids_by_tag(self, tag: str) -> Set[str]:
        """Finds all BCKG objects with a given tag.

        Args:
            tag: string representing the tag of interest

        Returns:
            a set of unique IDs of BCKG objects with given tag
        """
        query = "tags()"
        queryfilter = self._queryfilter("tag", tag)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        return set(jsonval["source"] for jsonval in x["value"])

    def list_tags_associated_to_entity(self, entity_id: str) -> Set[str]:
        """Returns the set of tags associated to a specific entity."""
        query = "tags()"
        queryfilter = self._queryfilter("source", entity_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        return set(jsonval["tag"] for jsonval in x["value"])

    @staticmethod
    def experiment_from_properties(
        xres,
        operations_map: Dict[str, List[domain.ExperimentOperation]],
        signals_map: Dict[str, List[domain.Signal]],
        samples_map: Dict[str, List[domain.Sample]],
    ) -> domain.Experiment:
        guid = xres["RowKey"]
        expt_type = xres["Type"]
        operations = operations_map.get(guid, [])
        signals = signals_map.get(guid, [])
        samples = samples_map.get(guid, [])
        if expt_type == "TypeIIs assembly":
            return domain.BuildExperiment(
                guid=guid,
                name=xres["Name"],
                notes=xres["Notes"],
                operations=operations,
                signals=signals,
                samples=samples,
            )
        elif expt_type == "Characterization":
            return domain.TestExperiment(
                guid=guid,
                name=xres["Name"],
                notes=xres["Notes"],
                operations=operations,
                signals=signals,
                samples=samples,
            )
        else:
            raise TypeError("Unknown Experiment Type encountered")

    def get_experiment_by_id(self, experiment_id: str) -> Optional[domain.Experiment]:  # type: ignore
        """Returns an experiment represented by its ID

        Args:
            experiment_id: experiment ID

        Returns:
            experiment with a given ID or None (if the experiment with a given ID doesn't exist)
        """
        print(f"Retrieving data for {experiment_id}")
        query = "experiments()"
        queryfilter = self._queryfilter("RowKey", experiment_id)
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        if len(x["value"]) == 0:
            return None
        else:
            xres = x["value"][0]
            operations = self.get_experiment_operations_by_expt_ids({experiment_id})
            signals = self.get_experiment_signals_by_expt_ids({experiment_id})
            samples = self.get_experiment_samples_by_expt_ids({experiment_id})
            return AzureConnection.experiment_from_properties(xres, operations, signals, samples)

    def get_experiments_from_guids(self, experiment_ids: Set[str]) -> List[domain.Experiment]:
        """Returns an experiment represented by its ID

        Args:
            experiment_id: experiment ID

        Returns:
            experiment with a given ID or None (if the experiment with a given ID doesn't exist)
        """
        query = "experiments()"
        query_experiments = [("RowKey", e_id) for e_id in experiment_ids]
        queryfilter = self._querydisjunction(query_experiments)
        res = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        valid_expt_ids = {xres["RowKey"] for xres in res["value"]}
        operations = self.get_experiment_operations_by_expt_ids(valid_expt_ids)
        signals = self.get_experiment_signals_by_expt_ids(valid_expt_ids)
        samples = self.get_experiment_samples_by_expt_ids(valid_expt_ids)
        experiments = [
            AzureConnection.experiment_from_properties(xres, operations, signals, samples) for xres in res["value"]
        ]
        return experiments

    # Include a function for Local-Cache too?
    def get_experiments_by_tags(self, tags: Iterable[str]) -> List[domain.Experiment]:
        guids = []
        for tag in tags:
            for guid in self.list_entity_ids_by_tag(tag):
                guids.append(guid)
        ids = set(guids)
        return self.get_experiments_from_guids(ids)

    def get_experiments_containing_tags(self, tags: Set[str]) -> List[domain.Experiment]:
        """Returns experiments that have *all* the tags specified."""
        if len(tags) == 0:
            return []
        else:
            # Find the entities that are associated with a particular tag.
            matching_to_one_tag: List[Set[str]] = [set(self.list_entity_ids_by_tag(tag)) for tag in tags]
            # Calculate the intersection
            guids: Set[str] = matching_to_one_tag[0].intersection(*matching_to_one_tag)
            return self.get_experiments_from_guids(guids)

    def get_tags(self, ids: Set[str]) -> Dict[str, Set[str]]:
        """Get all tags associated with the GUIDs specified in `ids`.
        This function returns a dictionary mapping the GUID to a `Set` of tags."""
        query = "tags()"
        sample_query = [("source", id) for id in ids]
        queryfilter = self._querydisjunction(sample_query)
        res = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        tag_map: Dict[str, Set[str]] = {}
        for xres in res["value"]:
            id = xres["source"]
            tag = xres["tag"]
            tag_map[id] = {tag} if id not in tag_map else tag_map[id].union(tag)
        return tag_map

    def get_reagents_by_barcode(self, barcodes: Set[str]) -> Set[domain.Reagent]:
        query = "reagents()"
        queryfilter = self._querydisjunction([("Barcode", barcode) for barcode in barcodes])
        x = self.query_table(HttpRequestMethod.GET, query, queryfilter)
        reagents = {AzureConnection.reagent_from_properties(xres) for xres in x["value"]}
        return reagents


def from_connection_string(connection_string: str, verbose: bool = True) -> AzureConnection:
    """Creates an AzureConnection from a connection string.

    Args:
        connection_string: connection string in the format "Key1=Value1;Key2=Value2;...". It must define the following
            keys: DefaultEndpointsProtocol, AccountName, AccountKey, BlobEndpoint, QueueEndpoint, TableEndpoint,
            FileEndpoint.
        verbose: controls the verbosity of the created connection
    """
    # Split the connection string into separate fields Key=Value, and then make it a dictionary
    attribs_raw: List[str] = connection_string.split(";")
    attribs: Dict[str, str] = dict()

    for attribute in attribs_raw:
        attribute_split = attribute.split("=", 1)
        if len(attribute_split) == 2:
            key, value = attribute_split
            attribs[key] = value

    return AzureConnection(
        default_endpoints_protocol=attribs["DefaultEndpointsProtocol"],
        account_name=attribs["AccountName"],
        account_key=attribs["AccountKey"],
        blob_endpoint=attribs["BlobEndpoint"],
        queue_endpoint=attribs["QueueEndpoint"],
        table_endpoint=attribs["TableEndpoint"],
        file_endpoint=attribs["FileEndpoint"],
        verbose=verbose,
    )
