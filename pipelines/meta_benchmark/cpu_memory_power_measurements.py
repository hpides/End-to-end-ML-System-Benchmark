import numpy as np
import time

import umlaut as eb
from meta_benchmark import bm
import argparse
import subprocess


parser = argparse.ArgumentParser(description="Umlaut benchmark configs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-m", "--memory", action="store_true", help="activate memory measurement", default=False)
parser.add_argument("-gm", "--gpumemory", action="store_true", help="activate gpu memory measurement", default=False)
parser.add_argument("-gt", "--gputime", action="store_true", help="activate gpu memory measurement", default=False)
parser.add_argument("-gp", "--gpupower", action="store_true", help="activate gpu memory measurement", default=False)
parser.add_argument("-t", "--time", action="store_true", help="activate time measurement", default=False)
parser.add_argument("-c", "--cpu", action="store_true", help="activate cpu measurement", default=False)
parser.add_argument("-g", "--gpu", action="store_true", help="activate gpu measurement", default=False)
parser.add_argument("-mf", "--memoryfreq", type=float, help="Interval for memory measurement", default=0.1)
parser.add_argument("-cf", "--cpufreq", type=float, help="Interval for cpu measurement", default=0.1)
parser.add_argument("-o", "--order", nargs="+", help="Specify the order of operations, multiple measurements of same kind possible.\n"
                                                     "Choose from: \"sleep\", \"sort\", \"mult\" and \"vw\"", required=True)
parser.add_argument("-r", "--repeat", type=int, help="How often to repeat measurements", default=1)
args = parser.parse_args()
config = vars(args)

metrics = []
types = []
if config["memory"]:
    metrics.append(eb.MemoryMetric('memory', interval=config["memoryfreq"]))
    types.append("memory")
if config["gpumemory"]:
    metrics.append(eb.GPUMemoryMetric('gpumemory', interval=config["memoryfreq"]))
    types.append("gpumemory")
if config["cpu"]:
    metrics.append(eb.CPUMetric('cpu', interval=config["cpufreq"]))
    types.append("cpu")
if config["gpu"]:
    metrics.append(eb.GPUMetric('gpu', interval=config["cpufreq"]))
    types.append("gpu")
if config["gpupower"]:
    metrics.append(eb.GPUPowerMetric('gpupower', interval=config["cpufreq"]))
    types.append("gpupower")
if config["gputime"]:    
    metrics.append(eb.GPUTimeMetric('gputime'))
    types.append("gputime")
if config["time"] or len(metrics) == 0:
    metrics.append(eb.TimeMetric('time'))
    types.append("time")


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
    if any(["gpu" in t for t in types]):
        print("Matrix Multiplying on the GPU")
        import torch
        # Create random matrices and move them to the GPU
        a = torch.rand((20000, 20000), device='cuda')
        b = torch.rand((20000, 20000), device='cuda')
        
        # Perform matrix multiplication on the GPU
        ab = torch.matmul(a, b)
        
        # Move result back to CPU and print (optional, here we'll print only a small part of it)
        ab_cpu = ab.cpu()
        print(ab_cpu[:5, :5])  
    else:
        print("Matrix Multiplying")
        a = np.random.random((10000, 10000))
        b = np.random.random((10000, 10000))
        ab = np.matmul(a, b)
        print(ab)


@eb.BenchmarkSupervisor(metrics, bm)
def vw_from_csv():
    print("Calculating VW")
    vw_main()
    print("Done")


def main():
    operation_dict = {"sleep": sleep, "sort": sorting, "mult": matrix_mult, "vw": vw_from_csv}
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

    subprocess.run(["umlaut-cli", "benchmark.db", "-u", uuid, "-t"]+types+["-d"]+types+["-p", "plotly"])


if __name__ == "__main__":
    main()
