from decorators import MeasureTime, MeasureMemorySamples, MeasureLatency, MeasureThroughput
from config import parser as config
import numpy as np
import sys


@MeasureTime(config['filepaths']['out_file'])
def print_this(string, times):
    for i in range(times):
        print(string)


@MeasureMemorySamples(config['filepaths']['out_file'], 0.1)
@MeasureTime(config['filepaths']['out_file'])
@MeasureLatency(config['filepaths']['out_file'], 40000000)
@MeasureThroughput(config['filepaths']['out_file'], 40000000)
def bloat(minsize, maxsize, step):
    a = None
    for i in range(minsize, maxsize, step):
        a = np.random.rand(i, i)
    return a


def main():
    print(bloat(0, 20000, 1000))


if __name__ == "__main__":
    main()
