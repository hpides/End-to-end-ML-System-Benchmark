import os
import sys
sys.path.insert(0, os.getcwd())
import package as pkg

bm = pkg.Benchmark('MNIST_benchmark.db', description="Complete MNIST MLP data benchmark")
