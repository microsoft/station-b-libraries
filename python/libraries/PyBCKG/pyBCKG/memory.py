# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import pyBCKG.domain as domain
from typing import Dict


class MemoryStorage:
    def __init__(self):
        self.parts = {}
        self.reagents: Dict[str, domain.Reagent] = {}
        self.cells: Dict[str, domain.Cell] = {}
        self.signals: Dict[str, domain.Signal] = {}

    def add_signal(self, signal):
        self.signals[signal.guid] = signal

    def add_part(self, part):
        self.parts[part.guid] = part

    def add_reagent(self, reagent):
        self.reagents[reagent.guid] = reagent

    def add_cell(self, cell):
        self.cells[cell.guid] = cell
