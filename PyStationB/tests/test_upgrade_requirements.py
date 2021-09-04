# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from scripts.upgrade_requirements import (
    edit_requirements_line,
    get_selected_versions,
    attach_specified_version,
    SUCCESS_KEY,
)


def test_edit_requirements_line():
    assert edit_requirements_line("foo", {}) == "foo"
    assert edit_requirements_line("name: PyStationB", {}) == "name: PyStationBUpgraded"
    assert edit_requirements_line("-r requirements.txt", {}) == "-r requirements_upgraded.txt"
    assert edit_requirements_line("foo==1.2.3", {}) == "foo"
    assert edit_requirements_line("foo==1.2.3", {"foo": "1.2.0"}) == "foo==1.2.0"


def test_get_selected_version():
    lines = f"blah blah\n{SUCCESS_KEY}foo-1.2.3 bar-baz-0.3\n"
    assert get_selected_versions(lines) == {"foo": "1.2.3", "bar-baz": "0.3"}


def test_attach_specified_version():
    sel_dct = {"foo": "1.2.3", "bar": "0.3"}
    reported = set()
    assert attach_specified_version("foo==1.0.0", sel_dct, reported) == "foo==1.2.3"
    assert attach_specified_version("bar==0.3", sel_dct, reported) == "bar==0.3"
    assert attach_specified_version("baz==1.0.0", sel_dct, reported) == "baz==1.0.0"
    assert reported == {"foo"}
