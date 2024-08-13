#!/bin/sh

python cpu_memory_power_measurements.py -o sleep sort sleep mult -t
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -c
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -m
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -c -m
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -c -m -g -gm -gt -gp