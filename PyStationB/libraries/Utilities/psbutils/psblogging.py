# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
"""
A subset of the useful functions in

https://github.com/microsoft/InnerEye-DeepLearning/blob/master/InnerEye/Common/common_util.py,

copied on 2020-10-15 and subsequently modified.
"""
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from psbutils.misc import check_is_any_of

LogLevel = Union[int, str]

logging_stdout_handler: Optional[logging.StreamHandler] = None
logging_to_file_handler: Optional[logging.StreamHandler] = None


def logging_to_stdout(log_level: LogLevel = logging.INFO) -> None:
    """
    Instructs the Python logging libraries to start writing logs to stdout up to the given logging level.
    Logging will use a timestamp as the prefix, using UTC.
    :param log_level: The logging level. All logging message with a level at or above this level will be written to
    stdout. log_level can be numeric, or one of the pre-defined logging strings (INFO, DEBUG, ...).
    """
    log_level = standardize_log_level(log_level)
    logger = logging.getLogger()
    # This function can be called multiple times, in particular in AzureML when we first run a training job and
    # then a couple of tests, which also often enable logging. This would then add multiple handlers, and repeated
    # logging lines.
    global logging_stdout_handler
    if not logging_stdout_handler:
        logging.info("Setting up logging to stdout.")
        # At startup, logging has one handler set, that writes to stderr, with a log level of 0 (logging.NOTSET)
        if len(logger.handlers) == 1:
            logger.removeHandler(logger.handlers[0])  # pragma: no cover
        logging_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        _add_formatter(logging_stdout_handler)
        logger.addHandler(logging_stdout_handler)
    logging.info(f"Setting logging level to {log_level}")
    logging_stdout_handler.setLevel(log_level)
    logger.setLevel(log_level)


# noinspection PyUnresolvedReferences
def standardize_log_level(log_level: LogLevel) -> int:  # pragma: no cover
    """
    :param log_level: integer or string (any casing) version of a log level, e.g. 20 or "INFO".
    :return: integer version of the level; throws if the string does not name a level.
    """
    if isinstance(log_level, str):
        log_level = log_level.upper()
        # noinspection PyProtectedMember
        check_is_any_of("log_level", log_level, logging._nameToLevel.keys())
        # noinspection PyProtectedMember
        return logging._nameToLevel[log_level]
    return log_level


def logging_to_file(file_path: Path) -> None:
    """
    Instructs the Python logging libraries to start writing logs to the given file.
    Logging will use a timestamp as the prefix, using UTC. The logging level will be the same as defined for
    logging to stdout.
    :param file_path: The path and name of the file to write to.
    """
    # This function can be called multiple times, and should only add a handler during the first call.
    global logging_to_file_handler
    if not logging_to_file_handler:  # pragma: no cover
        global logging_stdout_handler
        log_level = logging_stdout_handler.level if logging_stdout_handler else logging.INFO
        logging.info(f"Setting up logging with level {log_level} to file {file_path}")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(filename=str(file_path))
        _add_formatter(handler)
        handler.setLevel(log_level)
        logging.getLogger().addHandler(handler)
        logging_to_file_handler = handler


def disable_logging_to_file() -> None:
    """
    If logging to a file has been enabled previously via logging_to_file, this call will remove that logging handler.
    """
    global logging_to_file_handler
    if logging_to_file_handler:  # pragma: no cover
        logging_to_file_handler.close()
        logging.getLogger().removeHandler(logging_to_file_handler)
        logging_to_file_handler = None


# noinspection PyTypeHints
@contextmanager
def logging_only_to_file(file_path: Path, stdout_log_level: LogLevel = logging.ERROR) -> Generator:  # pragma: no cover
    """
    Redirects logging to the specified file, undoing that on exit. If logging is currently going
    to stdout, messages at level stdout_log_level or higher (typically ERROR) are also sent to stdout.
    Usage: with logging_only_to_file(my_log_path): do_stuff()
    :param file_path: file to log to
    :param stdout_log_level: mininum level for messages to also go to stdout
    """
    stdout_log_level = standardize_log_level(stdout_log_level)
    logging_to_file(file_path)
    global logging_stdout_handler
    if logging_stdout_handler is not None:
        original_stdout_log_level = logging_stdout_handler.level
        logging_stdout_handler.level = stdout_log_level  # type: ignore
        yield
        logging_stdout_handler.level = original_stdout_log_level
    else:
        yield
    disable_logging_to_file()


def _add_formatter(handler: logging.StreamHandler) -> None:  # pragma: no cover
    """
    Adds a logging formatter that includes the timestamp and the logging level.
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    # noinspection PyTypeHints
    formatter.converter = time.gmtime  # type: ignore
    handler.setFormatter(formatter)


@contextmanager
def logging_section(gerund: str) -> Generator:  # pragma: no cover
    """
    Context manager to print "**** STARTING: ..." and "**** FINISHED: ..." lines around sections of the log,
    to help people locate particular sections. Usage:
    with logging_section("doing this and that"):
       do_this_and_that()
    :param gerund: string expressing what happens in this section of the log.
    """
    from time import time

    logging.info("")
    msg = f"**** STARTING: {gerund} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")
    start_time = time()
    yield
    elapsed = time() - start_time
    logging.info("")
    if elapsed >= 3600:
        time_expr = f"{elapsed/3600:0.2f} hours"
    elif elapsed >= 60:
        time_expr = f"{elapsed/60:0.2f} minutes"
    else:
        time_expr = f"{elapsed:0.2f} seconds"
    msg = f"**** FINISHED: {gerund} after {time_expr} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")
