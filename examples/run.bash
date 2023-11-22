#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

cd 1
bash run.bash $1

cd ../2_source_location
bash run.bash $1

cd ../3_medium
bash run.bash $1

cd ../4
bash run.bash $1
