import data_preparation
import train
import test
from benchmarking import bm


def main():
    nn = data_preparation.data_preparation()
    model = train.train(nn[0], nn[1])
    test.test(model["model"], nn[2])
    bm.close()


if __name__ == "__main__":
    main()
