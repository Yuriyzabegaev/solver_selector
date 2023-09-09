#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
    "poro_coldstart_eps0.py"
    "poro_warmstart_eps0.py"
    "poro_coldstart_eps_big.py"
    "poro_warmstart_eps_big.py"

    "poro_coldstart_gp.py"
    "poro_warmstart_gp.py"
    "poro_warmstart_gp_no_exploration.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done

