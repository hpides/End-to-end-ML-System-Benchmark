import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd())
import package as pkg

def main():
    with pkg.Benchmark('test.db') as bm:
        @pkg.MeasureMemorySamples(bm, interval=0.1)
        @pkg.MeasureTime(bm)
        def bloat(minsize, maxsize, step):
            a = None
            for i in range(minsize, maxsize, step):
                a = np.random.rand(i, i)
            return a

        @pkg.MeasureConfusion(bm)
        def confuse():
            return {'TP': 1, 'FP': 1, 'TN': 1, 'FN': 1}

        print(bloat(0, 2000, 100))
        print(confuse())
    


if __name__ == "__main__":
    main()
