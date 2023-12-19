#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
    "thermal_coldstart_source_location.py"
    # "thermal_coldstart_source_location_no_expl.py"
    "thermal_warmstart_source_location.py"
    # "thermal_warmstart_source_location_no_expl.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target $i
    done
done

