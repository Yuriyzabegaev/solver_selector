#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass a number of how many times to execute each experiment. Example: \"bash run.bash 3\""
    echo "Exiting."
    exit 1
fi

num_repeats=$1

targets=(
    # "poro_coldstart_s.py"
    # "poro_coldstart_m.py"
    # "poro_coldstart_l.py"
    # "poro_warmstart_m_s.py"
    # "poro_warmstart_l_s.py"
    # "poro_warmstart_l_m.py"
    # "poro_warmstart_l_sm.py"
    
    "thermal_coldstart_m.py"
    "thermal_warmstart_m_s.py"
    # "thermal_coldstart_l.py"
    # "thermal_warmstart_l_sm.py"

    "thermal_warmstart_m_s_stacking.py"
    # "thermal_warmstart_l_sm_stacking.py"
)


for target in ${targets[@]}; do
    for ((i = 1; i <= $num_repeats; i++)); do
        echo
        echo $target
        python $target
    done
done

