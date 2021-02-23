import os
import sys
import package as pkg

sys.path.insert(0, os.getcwd())
bm = pkg.Benchmark('MNIST_benchmark.db', description="Complete MNIST MLP data benchmark")
