# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from pathlib import Path

from scripts.subrepositories_changed import add_subrepositories_depending_on


def test_add_subrepositories_depending_on():
    all_subrepos = set([path.name for path in Path(".").glob("*/*")])
    reqs = {"ABEX": ["Emukit"], "CellSignalling": ["ABEX", "Emukit"], "Malvern": ["ABEX"]}

    def req_getter(name):
        return reqs.get(name, [])

    assert add_subrepositories_depending_on({"ABEX"}, all_subrepos, req_getter) == {"ABEX", "CellSignalling", "Malvern"}
    assert add_subrepositories_depending_on({"Emukit"}, all_subrepos, req_getter) == {
        "Emukit",
        "ABEX",
        "CellSignalling",
        "Malvern",
    }
