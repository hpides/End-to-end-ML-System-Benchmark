import time
import numpy as np
from umlaut import Benchmark, BenchmarkSupervisor, MemoryMetric, CPUMetric

bm = Benchmark('hello_world.db', description="Database for the Github sample measurements")

bloat_metrics = {
    "memory": MemoryMetric('bloat memory', interval=0.1),
    "cpu": CPUMetric('bloat cpu', interval=0.1)
}


@BenchmarkSupervisor(bloat_metrics.values(), bm)
def bloat():
    a = []
    for i in range(1, 2):
        a.append(np.random.randn(*([10] * i)))
        time.sleep(5)
    print(a)


def main():
    bloat()
    bm.close()


if __name__ == "__main__":
    main()
