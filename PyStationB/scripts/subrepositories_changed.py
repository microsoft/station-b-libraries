# -------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# -------------------------------------------------------------------------------------------
import collections
import sys
from pathlib import Path
from typing import Callable, List, Set

from git import Repo

SUBREPOSITORY_PARENT_PATHS = [Path("projects"), Path("libraries")]
BASENAMES_TO_IGNORE = {".testmondata", "hashes.txt"}


def get_internal_requirements(name: str) -> List[str]:  # pragma: no cover
    result = []
    for path in SUBREPOSITORY_PARENT_PATHS:
        req_file = path / name / "internal_requirements.txt"
        if req_file.exists():
            with req_file.open() as f:
                for line in f:
                    dep = line.split("#", 1)[0].strip()
                    if dep:
                        result.append(dep)
    return result


def add_subrepositories_depending_on(
    changed: Set[str], all_subrepos: Set[str], requirement_getter: Callable[[str], List[str]]
) -> Set[str]:
    """
    Augments "changed" with other members of "all_subrepos" that directly or indirectly depend on the
    members of "changed", as listed in their internal_requirements.txt files.
    """
    if not changed:
        return changed  # pragma: no cover
    # Mapping from subrepo names to subrepos that directly depend on them.
    dependency_graph = collections.defaultdict(list)
    for name in all_subrepos:
        deps = requirement_getter(name)
        for dep in deps:
            if dep in all_subrepos:
                dependency_graph[dep].append(name)
    result = changed.copy()
    while True:
        additional = set()
        for name in result:
            additional.update(dependency_graph[name])
        additional.difference_update(result)
        if not additional:
            break  # we've converged
        result.update(additional)
    return result


def subrepositories_changed(all_if_master: bool = False) -> List[str]:  # pragma: no cover
    """
    Returns a list of the final name components of subrepositories that contain files that are different between the
    master branch and the current branch. Subrepositories are defined as the directories immediately under "projects"
    and "libraries".

    Example: if libraries/ABEX/foo/bar.py and projects/CellSignalling/bar/baz.py have changed, the result returned
    would be ["ABEX", "CellSignalling"].

    If the current branch *is* master, then all subrepository names (if all_if_master) or an empty list, is returned.

    "master" is tried as the name of the master branch, followed by "main" if that branch does not exist. If
    neither is found, which may be the case during an ADO build, we look at .git/FETCH_HEAD, which may show
    evidence of the master branch having been fetched, and if so will tell us its commit ID.
    """
    all_subrepos: Set[str] = set()
    for path in SUBREPOSITORY_PARENT_PATHS:
        for subrepo in path.glob("*"):
            if subrepo.is_dir():
                all_subrepos.add(subrepo.name)
    repo = Repo(".")
    master_branch_name = None
    for branch in repo.branches:
        if branch.name in ["master", "main"]:
            master_branch_name = branch.name
            break
    if master_branch_name is None:
        fh_path = Path(".git") / "FETCH_HEAD"
        if fh_path.exists():
            with fh_path.open() as fh:
                for line in fh.readlines():
                    if line.find("'master'") > 0 or line.find("'main'") > 0:
                        # master_branch_name is actually a commit in this case
                        master_branch_name = line.split(None, 1)[0]
                        sys.stderr.write(f"Setting master 'branch' name to commit {master_branch_name}\n")
                        break
    if master_branch_name is None:
        # Play safe: master branch not found, so assume all subrepos might have changed.
        sys.stderr.write("WARNING: could not find either a 'master' branch or a 'main' branch.\n")
        changed = all_subrepos
    else:
        changed = set()
        for diff in repo.index.diff(master_branch_name):
            for path in [Path(diff.a_path), Path(diff.b_path)]:
                parts = path.parts
                if (
                    len(parts) >= 2
                    and Path(parts[0]) in SUBREPOSITORY_PARENT_PATHS
                    and parts[1] in all_subrepos
                    and parts[-1] not in BASENAMES_TO_IGNORE
                ):
                    changed.add(parts[1])
        if changed:
            changed = add_subrepositories_depending_on(changed, all_subrepos, get_internal_requirements)
        elif all_if_master and current_commit_is_master(repo, master_branch_name):
            changed = all_subrepos
    # Remove subrepositories that appear to be submodules
    apparent_submodules = set(path.parent.name for path in Path(".").glob("*/*/.git"))
    result = [name for name in sorted(changed) if name not in apparent_submodules]
    return result


def current_commit_is_master(repo: Repo, master_branch_name: str) -> bool:  # pragma: no cover
    """
    :param repo: repository
    :param master_branch_name: "master", "main", or the hex commit ID thereof
    :return: whether the current branch (or commit) is the master one.
    """
    if repo.commit().hexsha == master_branch_name:  # type: ignore # auto
        return True  # current commit is master commit
    try:
        return repo.active_branch.name == master_branch_name
    except TypeError:  # no active branch, so assume we're not in master
        pass
    return False  # we're in another branch or commit than master


if __name__ == "__main__":
    print(" ".join(subrepositories_changed(all_if_master=True)))  # pragma: no cover
