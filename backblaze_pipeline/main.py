
import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd())
import package as pkg
from benchmarking import bm

from data_preparation import prepare_data
from train_and_test import train_and_test

def main():
    prepare_data()
    train_and_test()
    bm.close()

if __name__ == "__main__":
    main()
