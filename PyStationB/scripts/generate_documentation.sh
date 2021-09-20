#!/usr/bin/env bash

# This script generates Sphinx documentation from the contents of the "docs" directories at top level and
# under some (ideally all) subrepo directories (subdirectories of libraries/ and projects/). Documentation
# in each "docs/source" directory is manually written and stored in git; that in "docs/_modules" is automatically
# generated by the sphinx-apidoc command below.
#
# You should run this script when recommended at the end of initialize.sh, and whenever you find the HTML files that
# were generated last time it was run are out of date.
#
# The "make html" steps can take several minutes to run. They create files in docs/_build/html, which can
# be browsed at the URLs indicated in the output. They usually generate many lines of warnings, mostly
# inconsequential, so those outputs are stored in _sphinx_doc_warnings.txt in each docs/ directory.

ROOT=$PWD
SDW=_sphinx_doc_warnings.txt
if [ "$(basename $ROOT)" != "PyStationB" ]
then echo "Run this script from the PyStationB directory"
fi

for docs_dir in docs libraries/*/docs projects/*/docs
    do
    cd $docs_dir
    echo Generating documentation in $PWD
    MODULES=$(ls ../*/__init__.py 2>/dev/null | xargs -n 1 dirname)
    if [ -z "$MODULES" ]
    then echo WARNING: no directories with __init__.py found in $(dirname $PWD)
         continue
    fi
    rm -rf _modules
    sphinx-apidoc -f -o _modules/ $MODULES
    make clean
    URL_PATH=$(realpath . | sed 's@/mnt/c/@C:/@')
    echo 'Generating html; browse it at' file://$URL_PATH/_build/html/index.html
    make html > $SDW 2>&1
    NWARN=$(grep -whc "rst: WARNING:" $SDW)
    NERR=$(grep -whc "rst: ERROR:" $SDW)
    echo $PWD/$SDW has $NERR errors and $NWARN warnings
    cd $ROOT
    echo
    done