#!/usr/bin/env bash
# This script determines an appropriate setting for the PYTHONPATH variable to use in the PyStationB repository.
# It can only be called from a WSL/Linux shell. The recommended way of doing so is:
#    source scripts/set_pythonpath.py
# and then it will set PYTHONPATH for you. From a Windows command shell, you should instead call 
#    python scripts\get_pythonpath.py
# and then
#    set PYTHONPATH=...
# copying and pasting the value of PYTHONPATH that was printed.
#
# If an argument is given, it should be the basename of a subdirectory of "libraries" or "projects", e.g. ABEX.
# Then PYTHONPATH will contain only that subrepository and any listed in its "internal_requirements.txt" file.
# Otherwise, all subrepositories will be included.

if [ "$1" = "-q" ]
then ECHO=true
     shift
else ECHO=echo
fi

for d in . scripts
  do
  SCRIPT="$d"/get_pythonpath.py
  if [ -f "$SCRIPT" ]
  then break
  fi 
  done

if [ $# = 0 ]
then CMD="python $SCRIPT"
elif [ $# = 1 ]
then CMD="python $SCRIPT -s $1"
else echo Usage: $0 '[subrepo_name]'
     exit 1
fi
export PYTHONPATH=$($CMD)
$ECHO PYTHONPATH=$PYTHONPATH


