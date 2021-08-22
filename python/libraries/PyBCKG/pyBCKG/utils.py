# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
from datetime import datetime
from enum import Enum
from time import mktime as mktime
from wsgiref.handlers import format_date_time


class HttpRequestMethod(Enum):
    GET = "GET"
    POST = "POST"


def get_rfc_date_time():
    now = datetime.now()
    stamp = mktime(now.timetuple())
    rfc1123now = format_date_time(stamp)
    return rfc1123now


def remove_prefix(prefix, value):
    return value[len(prefix) :] if value.startswith(prefix) else value  # noqa
