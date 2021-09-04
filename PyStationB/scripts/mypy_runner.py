#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
from shutil import which
from typing import List, Optional, Set, Tuple


def run_mypy(files: List[str], mypy_executable_path: str, config_file: str) -> int:
    """
    Runs mypy on the specified files, printing whatever is sent to stdout (i.e. mypy errors).
    Because of an apparent bug in mypy, we run mypy in --verbose mode, so that log lines are printed to
    stderr. We intercept these, and assume that any files mentioned in them have been processed.
    We run mypy repeatedly on the files that were not mentioned until there are none remaining, or until
    no further files are mentioned in the logs.
    :param files: list of .py files to check
    :param mypy_executable_path: path to mypy executable
    :param config_file: path to mypy config file
    :return: maximum return code from any of the mypy runs
    """
    return_code = 0
    iteration = 1
    lines_to_print: Set[str] = set()  # We will print these in sorted order for easy reference.
    while files:
        dirs = sorted(set(os.path.dirname(file) or "." for file in files))
        print(f"Iteration {iteration}: running mypy on {len(files)} files in {len(dirs)} directories")
        # Set of files we are hoping to see mentioned in the mypy log.
        files_to_do = set(files)
        for index, dir in enumerate(dirs, 1):
            # Adding "--no-site-packages" might be necessary if there are errors in site packages,
            # but it may stop inconsistencies with site packages being spotted.
            command = [mypy_executable_path, f"--config-file={config_file}", "--verbose", dir]
            print(f"Processing directory {index:2d} of {len(dirs)}: {Path(dir).absolute()}")
            # We pipe stdout and then print it, otherwise lines can appear in the wrong order in builds.
            return_code = max(return_code, run_mypy_command(command, files_to_do, lines_to_print))
        if len(files_to_do) == len(files):  # pragma: no cover
            # If we didn't manage to discard any files, there's no point continuing. We hand the last few
            # files to mypy explicitly to give them the best chance of being processed.
            print(f"Processing final {len(files)} files")
            command = [mypy_executable_path, f"--config-file={config_file}", "--verbose"] + sorted(files)
            return_code = max(return_code, run_mypy_command(command, files_to_do, lines_to_print))
            break
        files = sorted(files_to_do)
        iteration += 1
    for line in sorted(lines_to_print):
        print(line)  # pragma: no cover
    sys.stdout.flush()
    return return_code if lines_to_print else 0


def is_innocent_message(line: str) -> bool:
    """
    Returns whether a line on stdout is "innocent", i.e. does not count as a real mypy error. We treat
    "skipping analyzing" lines as innocent, as these appear unpredictably and we aim to process all files anyway.
    """
    return (
        not line
        or line.startswith("Success: ")
        or line.find(": note: ") > 0
        or line.find(": error: Skipping analyzing ") > 0
        or line.startswith("Found ")
    )


def run_mypy_command(command: List[str], files_to_do: Set[str], lines_to_print: Set[str]) -> int:
    """
    Runs the (mypy) shell command given by "command" , which should process all the files in files_to_do.
    We process the output of the command, removing files from files_to_do when we find evidence they
    have been processed. When we find a log line that should be printed, we add it to lines_to_print.
    We return the returncode of the mypy command.
    """
    len1 = len(lines_to_print)
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout.split("\n"):
        if not is_innocent_message(line):  # pragma: no cover
            tokens = line.split(":")
            if len(tokens) < 2:
                print(line)  # pragma: no cover
            else:
                line = f"{Path.cwd() / tokens[0]}:{':'.join(tokens[1:])}"
                if line not in lines_to_print:
                    lines_to_print.add(line)
    # Remove from files_to_do every Python file that's reported as processed in the log.
    for line in process.stderr.split("\n"):
        tokens = line.split()
        if len(tokens) == 4 and tokens[0] == "LOG:" and tokens[1] == "Parsing":
            name = tokens[2]  # pragma: no cover
        elif len(tokens) == 7 and tokens[:4] == ["LOG:", "Metadata", "fresh", "for"]:
            name = tokens[-1]  # pragma: no cover
        else:
            continue  # pragma: no cover
        if name.endswith(".py"):
            if name.startswith("./") or name.startswith(".\\"):
                name = name[2:]  # pragma: no cover
            files_to_do.discard(name)
    len2 = len(lines_to_print)
    if len2 > len1:
        print(f"Found {len2 - len1} errors")  # pragma: no cover
    return process.returncode


def ini_file_lists(config_file: str) -> Tuple[List[str], List[str]]:  # pragma: no cover
    """
    Returns the values of "files" and "exclude" in the config file's "mypy" section.
    """
    config_path = Path(config_file)
    if not config_path.exists():
        return [], []
    config = ConfigParser()
    config.read(config_path)

    def get_section(name: str) -> List[str]:
        try:
            return [s for s in config["mypy"][name].split("\n") if s and not s.startswith("#") and Path(s).exists()]
        except KeyError:
            return []

    return get_section("files"), get_section("exclude")


def mypy_runner(arg_list: Optional[List[str]] = None) -> int:
    """
    Runs mypy on the files in the argument list, or every *.py file under the current directory if there are none.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="List of files to run mypy on. If not provided, run on current directory",
    )
    parser.add_argument(
        "-m",
        "--mypy",
        type=str,
        required=False,
        default=None,
        help="Path to mypy executable. If not provided, autodetect mypy executable.",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=False,
        default="mypy.ini",
        help="OptimizerConfig file to pass on to mypy.",
    )
    args = parser.parse_args(arg_list)
    if args.files:
        top_level_files = args.files
        exclude_files: List[str] = []
    else:  # pragma: no cover
        top_level_files, exclude_files = ini_file_lists(args.config_file)
        if not top_level_files:
            top_level_files = ["."]
    file_list: List[Path] = []
    for top in top_level_files:
        if top.endswith(".py"):
            file_list.append(Path(top))
        else:
            file_list.extend(Path(top).rglob("*.py"))  # pragma: no cover
    mypy = args.mypy or which("mypy")
    if not mypy:
        raise ValueError("Mypy executable not found.")  # pragma: no cover

    exclude_parts = [Path(exclude).parts for exclude in exclude_files]

    def should_exclude(path: Path) -> bool:
        for exc in exclude_parts:
            if path.parts[: len(exc)] == exc:  # pragma: no cover
                return True
        return False

    file_list = [path for path in file_list if not should_exclude(path)]
    return run_mypy(sorted(str(file) for file in file_list), mypy_executable_path=mypy, config_file=args.config_file)


if __name__ == "__main__":
    sys.exit(mypy_runner())  # pragma: no cover
