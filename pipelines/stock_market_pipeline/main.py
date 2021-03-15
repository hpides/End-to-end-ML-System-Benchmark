## from:
## https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f

import train
import test
import data_preparation
from benchmarking import bm


def main():
    df, model, X_train, y_train, sc = data_preparation.prepare_data()
    train_result = train.train(model, X_train, y_train)
    test.test(df, train_result["model"], sc)
    bm.close()


if __name__ == "__main__":
    main()
