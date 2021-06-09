import train_so2sat
import test_so2sat
from benchmarking import bm


def main():

    result = train_so2sat.train()
    test_so2sat.test(result)
    bm.close()


if __name__ == "__main__":
    main()
