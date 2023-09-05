#!/bin/bash

num_repeats=$1

targets=(
    "poro_coldstart_s.py"
    "poro_coldstart_m.py"
    "poro_coldstart_l.py"
    "poro_warmstart_m_s.py"
    "poro_warmstart_l_s.py"
    "poro_warmstart_l_m.py"
    "poro_warmstart_l_sm.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done

