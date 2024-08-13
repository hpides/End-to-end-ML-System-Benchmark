#!/bin/sh

python cpu_memory_power_measurements.py -o sleep mult sleep -t
python cpu_memory_power_measurements.py -o sleep mult sleep -t -c
python cpu_memory_power_measurements.py -o sleep mult sleep -t -m
python cpu_memory_power_measurements.py -o sleep mult sleep -t -c -m
python cpu_memory_power_measurements.py -o sleep mult sleep -t -c -m -g -gm -gt -gp