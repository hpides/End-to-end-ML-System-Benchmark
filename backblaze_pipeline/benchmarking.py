import os
import sys
import package as pkg

sys.path.insert(0, os.getcwd())
bm = pkg.Benchmark('backblaze_benchmark.db', description="Complete Backblaze data benchmark")
