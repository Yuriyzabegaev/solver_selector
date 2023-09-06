#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
    "poro_l0.py"
    "poro_l1.py"
    "poro_ldynamic.py"
    "thermal_cpr.py"
    "thermal_schur.py"
    "thermal_dynamic.py"
)

for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done
