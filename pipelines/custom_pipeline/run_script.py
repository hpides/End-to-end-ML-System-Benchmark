import sys
import os
import argparse
import umlaut
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Umlaut benchmark configs",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", "--script", type=str, help="Path to the script to execute", required=True)
    parser.add_argument("-m", "--memory", action="store_true", help="activate memory measurement", default=False)
    parser.add_argument("-gm", "--gpumemory", action="store_true", help="activate gpu memory measurement", default=False)
    parser.add_argument("-gt", "--gputime", action="store_true", help="activate gpu memory measurement", default=False)
    parser.add_argument("-gp", "--gpupower", action="store_true", help="activate gpu memory measurement", default=False)
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
    def execute_script(script_path):
        if not os.path.isfile(script_path):
            print(f"The file {script_path} does not exist.")
            return

        if not script_path.endswith('.py'):
            print("Please provide a Python script file.")
            return

        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)

        # Add the script's directory to PYTHONPATH
        sys.path.insert(0, script_dir)

        try:
            with open(script_path) as f:
                code = compile(f.read(), script_path, 'exec')
                exec(code, globals())
        except Exception as e:
            print(f"Error executing the script: {e}")
        finally:
            # Remove the directory from PYTHONPATH
            sys.path.pop(0)

    
    execute_script(config["script"])

    uuid = bm.uuid
    print("UUID", uuid)
    bm.close()

    subprocess.run(["umlaut-cli", "custom_script.db", "-u", uuid, "-t"]+types+["-d"]+types+["-p", "plotly"])

if __name__ == "__main__":
    main()