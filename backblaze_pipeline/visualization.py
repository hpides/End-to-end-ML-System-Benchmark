from datamodel import Measurement, BenchmarkMetadata
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc, asc

import pandas as pd
import seaborn as sn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import plot_confusion_matrix


def visualize(uuid):
    ## engine = create_engine('sqlite+pysqlite:///backblaze_benchmark.db')
    engine = create_engine('sqlite+pysqlite:///so2sat_benchmark.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df_dict = make_dataframe_from_database(uuid, session)
        plot_measurement_type(df_dict, "Memory")
        plot_confusion_matrix(df_dict)
        plot_time(df_dict)
        plot_throughput(df_dict)
        plot_latency(df_dict)

        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def make_dataframe_from_database(uuid, session):
    measure_query = session.query(Measurement.datetime,
                                  Measurement.description,
                                  Measurement.measurement_type,
                                  Measurement.value,
                                  Measurement.unit)
    measure_query = measure_query.filter_by(benchmark_uuid=uuid).order_by(asc(Measurement.datetime))

    df = pd.DataFrame(measure_query.all(), columns=["datetime", "description", "measurement_type", "value", "unit"])

    return df


def plot_measurement_type(df, measurement_type):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for description in df.description.unique():
        description_df = df.loc[(df.measurement_type == measurement_type) & (df.description == description)]
        ax.plot(description_df.datetime.values,
                description_df.value.values,
                label=description)

    ax.set_ylabel("MB of Memory")
    ax.set_xlabel("Time")
    plt.legend(loc=2)

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


def plot_confusion_matrix(df):
    values = df.loc[df.measurement_type == "Multiclass Confusion Matrix"].value.values
    classes = df.loc[df.measurement_type == "Multiclass Confusion Matrix Class"].value.values

    con_mat = []

    for i in range(len(classes)):
        con_mat_row = []
        for j in range(len(classes)):
            con_mat_row.append(int(values[j + i * len(classes)]))
        con_mat.append(con_mat_row)

    matrix = pd.DataFrame(con_mat, index=classes, columns=classes)
    plt.figure(figsize=(12, 8))
    plt.title("Confusion Matrix")
    sn.heatmap(matrix, annot=True)
    plt.show()


def plot_time(df):
    values = df.loc[df.measurement_type == "Time"].value.values
    label = df.loc[df.measurement_type == "Time"].description.values

    df = pd.DataFrame(
        dict(zip(label, values.astype(np.float))),
        index={"Runtime"}
    )

    ax = df.plot.barh(stacked=True)
    plt.title("Time spent in phases")
    plt.xlabel("Time in seconds")
    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        print(b)
        val = "{:.2f}".format(b.x1 - b.x0)
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))
    plt.show()


def plot_throughput(df):
    values = df.loc[df.measurement_type == "Throughput"].value.values
    label = df.loc[df.measurement_type == "Throughput"].description.values

    df = pd.DataFrame(
        dict(zip(label, values.astype(np.float))),
        index={"Throughput"}
    )
    ax = df.plot.barh()
    plt.title("Throughput")
    plt.xlabel("Seconds per entry")
    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        print(b)
        val = "{:.2f}".format(b.x1 - b.x0)
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))
    plt.show()


def plot_latency(df):
    values = df.loc[df.measurement_type == "Latency"].value.values
    label = df.loc[df.measurement_type == "Latency"].description.values

    df = pd.DataFrame(
        dict(zip(label, values.astype(np.float))),
        index={"Latency"}
    )
    ax = df.plot.barh()
    plt.title("Latency")
    plt.xlabel("Entries per second")
    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        print(b)
        val = "{:.2f}".format(b.x1 - b.x0)
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))
    plt.show()


if __name__ == "__main__":
    ## visualize('8a989821-f2e9-48da-ab41-36cb0fc6f580') ## Willi
    ## visualize("a8c2115c-0d6a-4cbc-ad47-445289d136fc") ## Jonas
    visualize("e9bc18ae-eb3e-4268-beef-0c6e8e43c00a")  ## Christian