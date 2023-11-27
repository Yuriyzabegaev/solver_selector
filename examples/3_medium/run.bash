#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
    # "poro_eps_default.py"
    # "poro_eps_big.py"
    # "poro_eps0.py"
    # "poro_gp.py"
    
    "thermal_eps_default.py"
    "thermal_eps_big.py"
    "thermal_eps0.py"
    # "thermal_gp.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done
