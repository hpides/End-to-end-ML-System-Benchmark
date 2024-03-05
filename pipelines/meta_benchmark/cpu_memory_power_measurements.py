import numpy as np
import time

import umlaut as eb
from meta_benchmark import bm
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Umlaut benchmark configs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--memory", action="store_true", help="activate memory measurement", default=True)
parser.add_argument("-t", "--time", action="store_true", help="activate time measurement", default=True)
parser.add_argument("-c", "--cpu", action="store_true", help="activate cpu measurement", default=True)
parser.add_argument("-mf", "--memoryfreq", type=float, help="Interval for memory measurement", default=0.1)
parser.add_argument("-cf", "--cpufreq", type=float, help="Interval for cpu measurement", default=0.1)
parser.add_argument("-o", "--order", nargs="+", help="Specify the order of operations, multiple measurements of same kind possible.\n"
                                                     "Choose from: \"sleep\", \"sort\" and \"mult\"", required=True)
parser.add_argument("-r", "--repeat", type=int, help="How often to repeat measurements", default=1)
args = parser.parse_args()
config = vars(args)

metrics = []
if config["memory"]:
    metrics.append(eb.MemoryMetric('memory', interval=config["memoryfreq"]))
if config["cpu"]:
    metrics.append(eb.CPUMetric('cpu', interval=config["cpufreq"]))
if config["time"] or len(metrics) == 0:
    metrics.append(eb.TimeMetric('time'))


@eb.BenchmarkSupervisor(metrics, bm)
def sorting():
    print("Sorting.")
    array = np.random.randint(low=0, high=1000, size=125000000)  # 12500000 = 100 MB
    print("Memory size of numpy array in bytes:", array.size * array.itemsize)
    array.sort()
    print(array[0])


@eb.BenchmarkSupervisor(metrics, bm)
def sleep():
    print("Sleeping.")
    time.sleep(10)
    print("Done")


@eb.BenchmarkSupervisor(metrics, bm)
def matrix_mult():
    print("Matrix Multiplying")
    a = np.random.random((10000, 10000))
    b = np.random.random((10000, 10000))
    ab = np.matmul(a, b)
    print(ab)


def main():
    operation_dict = {"sleep": sleep, "sort": sorting, "mult": matrix_mult}
    for i in range(config["repeat"]):
        print("Now running run " + str(i+1) + " of " + str(config["repeat"]))
        for operation in config["order"]:
            if operation in operation_dict:
                operation_dict[operation]()
            else:
                raise Exception("Unknown operation.")
    uuid = bm.uuid
    print(uuid)
    bm.close()

    subprocess.run(["umlaut-cli", "benchmark.db", "-u", uuid, "-t", "time" "memory", "cpu", "-d", "time", "memory", "cpu", "-p", "text"])


if __name__ == "__main__":
    main()
