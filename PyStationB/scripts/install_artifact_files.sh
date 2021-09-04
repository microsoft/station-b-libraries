#!/bin/bash

# Run this script on a drop.zip file created during a build when figureNNN.png files are created by
# failing tests on plotting code. For full instructions, see
#   libraries/Utilities/psbutils/filecheck.py

SCRIPT=$(dirname $0)/../libraries/Utilities/psbutils/install_artifact_files.py
if [ ! -e "$SCRIPT" ]
then echo File not found: "$SCRIPT"
     exit -1
fi
python "$SCRIPT" "$@"
