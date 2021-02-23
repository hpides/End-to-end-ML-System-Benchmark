import os
import sys
import package as pkg

sys.path.insert(0, os.getcwd())
bm = pkg.Benchmark('so2sat_benchmark.db', description="So2Sat image prediction benchmark")
