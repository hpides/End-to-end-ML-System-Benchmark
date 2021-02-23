from datamodel import Measurement
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import plot_confusion_matrix


def visualize(uuids, database_file):
    engine = create_engine(f'sqlite+pysqlite:///{database_file}')
    # engine = create_engine('sqlite+pysqlite:///so2sat_benchmark.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df_dict = make_dataframe_from_database_mult(uuids, session)
        plot_TTA(df_dict)
        plot_measurement_type(df_dict, "Memory (psutil)")
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
    measure_query = session.query(Measurement.benchmark_uuid,
                                  Measurement.datetime,
                                  Measurement.description,
                                  Measurement.measurement_type,
                                  Measurement.value,
                                  Measurement.unit)
    measure_query = measure_query.filter_by(benchmark_uuid=uuid).order_by(asc(Measurement.datetime))

    df = pd.DataFrame(measure_query.all(), columns=["uuid", "datetime", "description", "measurement_type", "value",
                                                    "unit"])

    return df


# visualize multiple uuids, also works with only a single one
def make_dataframe_from_database_mult(uuids, session):
    measure_query = session.query(Measurement.benchmark_uuid,
                                  Measurement.datetime,
                                  Measurement.description,
                                  Measurement.measurement_type,
                                  Measurement.value,
                                  Measurement.unit)

    query = measure_query.filter_by(benchmark_uuid=uuids[0])
    for uuid in range(1, len(uuids)):
        temp_query = measure_query.filter_by(benchmark_uuid=uuids[uuid])
        query = query.union(temp_query)
    query = query.order_by(asc(Measurement.datetime))

    df = pd.DataFrame(query.all(), columns=["uuid", "datetime", "description", "measurement_type", "value", "unit"])

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

    
def plot_TTA(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    tta_df = df.loc[(df.measurement_type == "TTA")]

    for uuid in tta_df.uuid.unique():
        tta_uuid_df = tta_df.loc[(df.uuid == str(uuid))]
        epochs = []
        for i in range(len(tta_uuid_df.value.values)):
            epochs.append(i+1)

        ax.plot(epochs,
                tta_uuid_df.value.values.astype(float),
                label=uuid)

    plt.legend(loc=2)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Total Epochs in training")
    plt.title("Time (Epochs) to Accuracy")

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


# currently need to cutoff first value due to first value always being 0
# needs to be changed sometime in decorators.py
def plot_energy(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    currentTimestamp = 0
    for description in df.description.unique():
        description_df = df.loc[(df.measurement_type == "Energy") & (df.description == description)]
        if len(description_df) != 0:
            duration = round((description_df.datetime.values[len(description_df.datetime.values) - 1] -
                              description_df.datetime.values[0]) / np.timedelta64(1, 's'))
            timevalues = []
            for i in range(len(description_df.datetime.values)):
                timevalues.append(len(description_df.datetime.values)*i/duration + currentTimestamp)
            currentTimestamp = timevalues[len(timevalues)-1]
            ax.plot(timevalues[1:len(description_df.datetime.values)],
                    description_df.value.values.astype(float)[1:len(description_df.datetime.values)],
                    label=description)

    ax.set_ylabel("mJ of energy usage")
    ax.set_xlabel("Time in seconds")
    plt.legend(loc=2)
    plt.title("Energy consumption")

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
    visualize('aca3920c-2a79-4a6e-bb88-62b2c382e27c')   ## Willi
    # visualize("a8c2115c-0d6a-4cbc-ad47-445289d136fc") ## Jonas
    # visualize("e9bc18ae-eb3e-4268-beef-0c6e8e43c00a") ## Christian
