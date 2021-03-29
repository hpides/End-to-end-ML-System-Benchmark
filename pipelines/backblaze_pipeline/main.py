import os
import sys

from benchmarking import bm
from data_preparation import prepare_data
from train_and_test import train_and_test

def main():
    try:
        prepare_data()
        train_and_test()
    finally:
        bm.close()

if __name__ == "__main__":
    main()
