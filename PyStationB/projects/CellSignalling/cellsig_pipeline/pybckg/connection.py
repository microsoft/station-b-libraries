# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""

Exports:
    WetLabAzureConnection, a child class of pyBCKG's AzureConnection, tailored to our needs
    create_connection, a factory method for WetLAbAzureConnection
"""
import datetime
import io
import os
import time
from typing import Iterable, List, Set, Tuple

import pandas as pd
import pyBCKG.azurestorage.api as api
import pyBCKG.domain as domain

N_TRIES: int = 3  # Maximal number of iterations when we try to download a timeseries file from the blob storage.


def _is_dataframe_valid(df: pd.DataFrame) -> bool:  # pragma: no cover
    """Verifies if data frame `df` represents timeseries.

    Returns:
        true iff the data frame is valid

    Note:
        This is a very simple heuristic, used to filter out non-existent files from the Azure blob storage. It doesn't
        catch more sophisticated issues.
    """
    return ("time" in df.columns) and (len(df) > 5)


class WetLabAzureConnection(api.AzureConnection):
    """Our internal version of the Azure Connection, extending the pyBCKG's one by several useful utilities, suitable
    for our workflow.

    Additional methods:
        get_timeseries_from_sample, returns a data frame with time series for a given sample
        experiment_ids_in_a_track, helps to retrieve the whole experimental track, associated to a set of tags
    """

    def get_timeseries_from_sample(self, sample: domain.Sample) -> pd.DataFrame:  # pragma: no cover
        """Tries to get the time series data frame for a given sample. This method is intended to be robust with respect
        to non-existent files, trying to download them a few times before an exception is thrown.

        Returns:
            a time series data frame as stored in BCKG (column keys are "time" and GUIDs)

        Raises:
            FileNotFoundError, if a valid data frame does not exist
        """
        file_ids = self.list_file_ids_of(sample.guid)
        assert len(file_ids) > 0, f"At least one file ID must be associated to sample {sample.guid}."

        for n_try in range(1, 1 + N_TRIES):
            for file_id in file_ids:
                raw_response: io.StringIO = self.get_timeseries_file(file_id)
                data: pd.DataFrame = pd.read_csv(raw_response)  # type: ignore # auto

                if _is_dataframe_valid(data):
                    return data
                else:
                    print(f"For sample {sample.guid} the file {file_id} is not valid: {raw_response.read()}.")
                    time.sleep(0.2)

            print(f"Retrieving at {n_try}. iteration was not successful for sample {sample.guid}.")
            time.sleep(1.0)

        raise FileNotFoundError(f"Retrieving the timeseries file for sample {sample.guid} unsuccessful.")

    def _retrieve_experiment_ids_by_tags(self, tags: Iterable[str]) -> Set[str]:  # pragma: no cover
        """The experiment IDs that have *all* `tags` specified.

        TODO: It seems that the new version of pyBCKG implements this utility.
        """
        # Find the entities that are associated with one tag.
        matching_to_one_tag: List[Set[str]] = [set(self.list_entity_ids_by_tag(tag)) for tag in tags]

        # Calculate the intersection
        answer: Set[str] = matching_to_one_tag[0].intersection(*matching_to_one_tag) if matching_to_one_tag else set()

        return answer

    @staticmethod
    def _find_last_operation(
        experiment_operations: Iterable[domain.ExperimentOperation],
    ) -> datetime.datetime:  # pragma: no cover
        """Returns the datetime of chronologically latest experiment operation."""
        return sorted(exp_op.timestamp for exp_op in experiment_operations)[-1]

    def _sort_experiment_ids_chronologically(self, experiment_ids: Iterable[str]) -> List[str]:  # pragma: no cover
        timestamped_ids: List[Tuple[datetime.datetime, str]] = [
            (self._find_last_operation(self.get_experiment_operations(exp_id)), exp_id) for exp_id in experiment_ids
        ]

        return [exp_id for _, exp_id in sorted(timestamped_ids)]

    def experiments_ids_in_a_track(self, tags: Iterable[str]) -> List[str]:  # pragma: no cover
        """Returns the IDs of experiments matching to *all* the tags. They are returned in the chronological order."""
        return self._sort_experiment_ids_chronologically(self._retrieve_experiment_ids_by_tags(tags))


def create_connection(conn_string: str) -> WetLabAzureConnection:  # pragma: no cover
    """Creates a wet-lab connection object retrieving the connection string from an environmental variable.

    Example:
        if the connection string is stored in the environmental variable 'BCKG_PRODUCTION_CONNECTION_STRING', pass
        this name as the argument to this function
    """
    key = os.environ[conn_string]
    template: api.AzureConnection = api.from_connection_string(key)

    return WetLabAzureConnection(
        default_endpoints_protocol=template.default_endpoints_protocol,
        account_name=template.account_name,
        account_key=template.account_key,
        blob_endpoint=template.blob_endpoint,
        queue_endpoint=template.queue_endpoint,
        table_endpoint=template.table_endpoint,
        file_endpoint=template.file_endpoint,
        verbose=True,
    )
