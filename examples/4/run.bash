#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
#    "thermal_many_solvers.py"
#    "thermal_many_solvers_gp.py"
#    "thermal_random.py"
    "thermal_coldstart_source_location.py"
    "thermal_warmstart_source_location.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done
