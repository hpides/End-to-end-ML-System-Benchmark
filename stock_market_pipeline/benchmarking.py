import os
import sys
sys.path.insert(0, os.getcwd())
import package as pkg

bm = pkg.Benchmark('stock_market_benchmark.db', description="Stock Market Prediction")
