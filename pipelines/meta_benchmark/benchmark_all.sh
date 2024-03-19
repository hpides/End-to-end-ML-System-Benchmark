#!/bin/sh

python cpu_memory_power_measurements.py -o sleep -t
python cpu_memory_power_measurements.py -o sleep -t -c
python cpu_memory_power_measurements.py -o sleep -t -m
python cpu_memory_power_measurements.py -o sleep -t -c -m

python cpu_memory_power_measurements.py -o sleep sort sleep -t
python cpu_memory_power_measurements.py -o sleep sort sleep -t -c
python cpu_memory_power_measurements.py -o sleep sort sleep -t -m
python cpu_memory_power_measurements.py -o sleep sort sleep -t -c -m

python cpu_memory_power_measurements.py -o sleep mult sleep -t
python cpu_memory_power_measurements.py -o sleep mult sleep -t -c
python cpu_memory_power_measurements.py -o sleep mult sleep -t -m
python cpu_memory_power_measurements.py -o sleep mult sleep -t -c -m

python cpu_memory_power_measurements.py -o sleep sort sleep mult -t
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -c
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -m
python cpu_memory_power_measurements.py -o sleep sort sleep mult -t -c -m

python cpu_memory_power_measurements.py -o vw -t
python cpu_memory_power_measurements.py -o vw -t -c
python cpu_memory_power_measurements.py -o vw -t -m
python cpu_memory_power_measurements.py -o vw -t -c -m
