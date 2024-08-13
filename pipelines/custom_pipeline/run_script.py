import sys
import os
import argparse
import umlaut
import subprocess
import psutil
import time

        
def main():
    parser = argparse.ArgumentParser(description="Umlaut benchmark configs",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-cmd", "--command", type=str, help="Command to execute", required=True)
    parser.add_argument("-folder", "--execution_folder", type=str, help="Where to execute the command", required=True)
    parser.add_argument("-m", "--memory", action="store_true", help="activate memory measurement", default=False)
    parser.add_argument("-gm", "--gpumemory", action="store_true", help="activate gpu memory measurement", default=False)
    parser.add_argument("-gt", "--gputime", action="store_true", help="activate gpu time measurement", default=False)
    parser.add_argument("-gp", "--gpupower", action="store_true", help="activate gpu power measurement", default=False)
    parser.add_argument("-t", "--time", action="store_true", help="activate time measurement", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", help="activate cpu measurement", default=False)
    parser.add_argument("-g", "--gpu", action="store_true", help="activate gpu measurement", default=False)
    parser.add_argument("-mf", "--memoryfreq", type=float, help="Interval for memory measurement", default=0.1)
    parser.add_argument("-cf", "--cpufreq", type=float, help="Interval for cpu measurement", default=0.1)
    args = parser.parse_args()
    config = vars(args)

    metrics = []
    types = []
    if config["memory"]:
        metrics.append(umlaut.MemoryMetric('memory', interval=config["memoryfreq"]))
        types.append("memory")
    if config["gpumemory"]:
        metrics.append(umlaut.GPUMemoryMetric('gpumemory', interval=config["memoryfreq"]))
        types.append("gpumemory")
    if config["cpu"]:
        metrics.append(umlaut.CPUMetric('cpu', interval=config["cpufreq"]))
        types.append("cpu")
    if config["gpu"]:
        metrics.append(umlaut.GPUMetric('gpu', interval=config["cpufreq"]))
        types.append("gpu")
    if config["gpupower"]:
        metrics.append(umlaut.GPUPowerMetric('gpupower', interval=config["cpufreq"]))
        types.append("gpupower")
    if config["gputime"]:    
        metrics.append(umlaut.GPUTimeMetric('gputime'))
        types.append("gputime")
    if config["time"] or len(metrics) == 0:
        metrics.append(umlaut.TimeMetric('time'))
        types.append("time")

    bm = umlaut.Benchmark('custom_script.db', description="Benchmark custom scripts.")

    @umlaut.BenchmarkSupervisor(metrics, bm)
    def execute_command(command, dir):
        # Change the working directory to the script's directory if provided
        original_dir = os.getcwd()
        os.chdir(dir)

        print(command)
        # Print the current directory
        print(f"Current directory: {os.getcwd()}")
        process = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    
        pid = process.pid
        for metric in metrics:
            if hasattr(metric, "process"):
                metric.process = psutil.Process(pid)
                metric.timestamps = []
                metric.measurements = []
                print("switched")
                
        stdout, stderr = process.communicate()

        # Print the subprocess output
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        
        os.chdir(original_dir)

    execute_command(config["command"], config["execution_folder"])

    uuid = bm.uuid
    print("UUID", uuid)
    bm.close()

    subprocess.run(["umlaut-cli", "custom_script.db", "-u", uuid, "-t"] + types + ["-d"] + types + ["-p", "plotly"])

if __name__ == "__main__":
    main()
