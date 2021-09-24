#!/usr/bin/env bash

# Run all tests, or a subset of them if there are arguments.
#   -t extra_test_dir: run tests in extra_test_dir (under each subrepo) as well as tests/
#   -f: "full" - i.e. run all tests rather than just those mandated by testmon
#   additional arguments: subrepo names, e.g. ABEX, or "." for top level. Default is all.

TEST_DIRS=tests
FULL=0
while getopts 'ft:' OPTION
  do
  case $OPTION in
    t) TEST_DIRS="$TEST_DIRS $OPTARG"
        ;;
    f) FULL=1
       ;;
  esac
  done
shift $(($OPTIND-1))
SUBREPO_NAMES="$*"

# Set up OS-dependent values
if [ -n "$(which pytest)" ]
then TESTMON_ENV=Linux
     PYTEST=pytest
     AUTO="-n=auto"
elif [ -n "$(which pytest.exe)" ]
then TESTMON_ENV=Windows
     PYTEST=pytest.exe
     # Running with xdist (-n=auto) frequently causes node crashes under Windows. This wastes more time than
     # running the tests slowly, as the whole job including the Conda install then has to be redone.
     AUTO=""
else echo $0: cannot find either pytest or pytest.exe
     exit 1
fi

function run_tests_in_dir {
     # Argument: a subrepository path relative to $ROOT, e.g. libraries/ABEX.
     # Emukit is a submodule so we don't run its tests. And not all subrepos have to have tests/ dirs.
     if [ $(basename "$1") = "Emukit" -o ! -d $1/tests ]
     then return
     fi
     if [ $(basename "$1") = "GlobalPenalisation" -o ! -d $1/tests ]
     then return
     fi
     # Set PYTHONPATH to have the right visibility for this subrepo.
     if [ "$1" = "." ]
     then export PYTHONPATH=$PWD
     else source scripts/set_pythonpath.sh "$(basename $1)"
     fi
     cd "$1"
     EXISTING=""
     for d in $TEST_DIRS
       do
       if [ -d $d ]
       then EXISTING="$EXISTING $d"
       fi
       done
     if [ -z "$EXISTING" ]
     then
       echo No test directories found in $PWD
     else
       if [ $FULL = 1 ]
       then CMD="$PYTEST -vv --cov $AUTO --timeout=2 --durations=10 $EXISTING"
       else CMD="$PYTEST -vv --testmon --testmon-env $TESTMON_ENV --timeout=2 --durations=10 $EXISTING"
       fi
       echo Running: $CMD
       $CMD
       RC=$?
       # RC is the return code from the $PYTEST command. If it's 5, this means no tests were run, which is
       # fine, so we switch it to 0.
       if [ $RC = 5 ]
       then RC=0
       fi
       # RET is the return code for the whole script. We set it to the maximum RC value found.
       if [ $RET -lt $RC ]
       then RET=$RC
       fi
     fi
     cd "$ROOT"
}

THIS=$(realpath $0)
HERE=$(dirname $THIS)
ROOT=$(dirname $HERE)
cd $ROOT

if [ -z "$SUBREPO_NAMES" ]
then SUBREPOS=$(echo . libraries/* projects/*)
else SUBREPOS=""
     for name in $SUBREPO_NAMES
        do
        if [ $name = . ]
        then SUBREPOS="$SUBREPOS ."
        else SUBREPOS="$SUBREPOS [lp]*s/$name"
        fi
        done
fi

RET=0
for subrepo in $SUBREPOS
    do
    run_tests_in_dir $subrepo
    done
if [ $RET != 0 ]
then echo CHECK THE OUTPUT ABOVE FOR ERRORS
fi
exit $RET