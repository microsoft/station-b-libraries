#!/usr/bin/env bash

# Sorts imports by running isort on most .py files under the current directory, with line length 120.
# Excluded files: anything under Emukit; __init__.py files; and anything where we call matplotlib.use("Agg")
# or plt.switch_backend("Agg"), because the import order needs to be non-standard in those.

if [ -z "$(which isort)" ]
then pip install isort[requirements_deprecated_finder]
fi
find * -name Emukit -prune -o -name __init__.py -prune -o -name '*.py' | \
  xargs egrep -wL '(use|switch_backend)\("Agg"\)' | \
  xargs isort -w 120


