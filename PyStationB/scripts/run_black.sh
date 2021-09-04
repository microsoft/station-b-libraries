#!/usr/bin/env bash
# Runs "black" code formatter on all .py files under the current directory, avoiding Emukit.
if [ -z "$(which black)" ]
then pip install black
fi
find * -name Emukit -prune -o -name '.*' -prune -o -name '*.py' -print | sort | xargs black
