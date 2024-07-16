#!/bin/sh

python cpu_memory_power_measurements.py -o sleep sort sleep -t
python cpu_memory_power_measurements.py -o sleep sort sleep -t -c
python cpu_memory_power_measurements.py -o sleep sort sleep -t -m
python cpu_memory_power_measurements.py -o sleep sort sleep -t -c -m
python cpu_memory_power_measurements.py -o sleep sort sleep -t -c -m -g -gm -gt -gp
