import numpy as np

from e2ebench import Benchmark, ConfusionMatrixTracker

def main():
    bm = Benchmark('sample_db_file.db')
    conf_mat = np.arange(9).reshape((3,3))
    labels = ['foo', 'bar', 'baz']
    ConfusionMatrixTracker(bm).track(conf_mat, labels, 'foobar')

if __name__ == "__main__":
    main()