#!/usr/bin/python3

from pathlib import Path
import pandas as pd
import numpy as np
import typer
# from kai_benchmark import bm
import umlaut as eb
# from line_simplification import vw_line


#@eb.BenchmarkSupervisor([eb.PowerMetric('end-to-end power'), eb.TimeMetric('end-to-end runtime'), eb.MemoryMetric('end-to-end memory', interval=1), eb.CPUMetric('end-to-end CPU us', interval=1)], bm)
#@eb.BenchmarkSupervisor([eb.TimeMetric('end-to-end runtime'),eb.MemoryMetric('end-to-end memory', interval=0.01), eb.CPUMetric('end-to-end CPU us', interval=0.01)], bm)
def main(
    # source_file: Path = typer.Option(..., "--source", help="File path to source file"),
    # min_points: int = typer.Option(2, "--min", help="min_points"),
    # max_points: int = typer.Option(18, "--max", help="max_points"),
    # tolerance: float = typer.Option(0.0001, "--tol", help="tolerance"),
    # source_file: Path = Path("data/PCF1_tdms_group_0.csv"),
    # source_file: Path = Path("data/LAH26/LAH26_1.csv"),
    min_points: int = 2,
    max_points: int = 18,
    tolerance: float = 0.0001
):
    for i in range(9421):
        source_file = Path("data/LAH26/LAH26_" + str(i) + ".csv")
        process(source_file, min_points, max_points, tolerance)


def process(csv_file: Path, min_points: int, max_points: int, tolerance: float):
    data = pd.read_csv(csv_file, header=None, names=["Id", "Vds", "Time"])

    p_tot = np.abs(data["Id"] * data["Vds"])
    time = np.arange(p_tot.shape[0]) * 0.000004

    power_over_time = pd.DataFrame(data=time, index=None, columns=["time"])
    power_over_time["power"] = p_tot

    # reduced_idxs = vw_line(data=power_over_time, max_points=max_points, min_points=min_points, tolerance=tolerance)
    # print(reduced_idxs)
    print("PROBLEM")
    print(power_over_time.head())


if __name__ == "__main__":
    # typer.run(main)
    main()
