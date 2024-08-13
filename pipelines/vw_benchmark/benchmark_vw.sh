#!/bin/sh

python ../meta_benchmark/cpu_memory_power_measurements.py -o vw -t
python ../meta_benchmark/cpu_memory_power_measurements.py -o vw -t -c
python ../meta_benchmark/cpu_memory_power_measurements.py -o vw -t -m
python ../meta_benchmark/cpu_memory_power_measurements.py -o vw -t -c -m
