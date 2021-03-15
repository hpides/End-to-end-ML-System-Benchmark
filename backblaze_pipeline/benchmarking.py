import os
import sys

sys.path.insert(0, os.getcwd())
import package as pkg


bm = pkg.Benchmark('backblaze_benchmark.db', description="Complete Backblaze data benchmark")
