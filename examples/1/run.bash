#!/bin/bash

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
