from datamodel import Measurement
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def visualize(uuids, database_file):
    engine = create_engine(f'sqlite+pysqlite:///{database_file}')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df = make_dataframe_from_database(uuids, session)
        metrics = df.measurement_type.unique()

        for metric in metrics:
            if metric == "Multiclass Confusion Matrix Class":           # helper class for MCCM, not a metric on its own
                continue
            for uuid in uuids:
                filtered_df = df.loc[(df.measurement_type == metric) & (df.uuid == uuid)]
                if len(filtered_df) != 0:
                    metrics_dict[metric](df)
                    break

        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# visualize multiple uuids, also works with only a single one
def make_dataframe_from_database(uuids, session):

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


def plot_time_based_graph(df, measurement_type, title, xlabel, ylabel):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    filtered_df = df.loc[(df.measurement_type == measurement_type)]

    for uuid in filtered_df.uuid.unique():
        filtered_uuid_df = filtered_df.loc[(df.uuid == uuid)]
        lastTimestamp, lastValue = 0, 0
        for description in df.description.unique():
            description_df = filtered_uuid_df.loc[df.description == description]
            if len(description_df) != 0:
                dates = description_df.datetime.values
                duration = round((dates[len(dates) - 1] - dates[0]) / np.timedelta64(1, 's'))
                if duration == 0:
                    duration = 1.0
                values = description_df.value.values.astype(float)
                timestamps = []
                if lastTimestamp != 0:
                    timestamps.append(lastTimestamp)
                    lastTimestamp += 1
                    values = np.insert(values, 0, lastValue)
                for i in range(len(dates)):
                    timestamps.append(len(dates)*i/duration + lastTimestamp)
                lastTimestamp = timestamps[len(timestamps)-1]
                lastValue = values[-1]

                ax.plot(timestamps,
                        values,
                        label=(uuid + ": " + description))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.legend(loc=2)
    plt.title(title)

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


def plot_memory(df):

    plot_time_based_graph(df, "Memory", "Memory usage", "Time in seconds", "MB used")


def plot_energy(df):
    plot_time_based_graph(df, "Energy", "Power consumption", "Time in seconds", "mJ of energy usage")


def plot_TTA(df):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    tta_df = df.loc[(df.measurement_type == "TTA")]

    for uuid in tta_df.uuid.unique():
        tta_uuid_df = tta_df.loc[(df.uuid == uuid)]
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


def plot_confusion_matrix(df):

    for uuid in df.uuid.unique():

        values = df.loc[(df.measurement_type == "Multiclass Confusion Matrix") & (df.uuid == uuid)].value.values
        if len(values) == 0:
            break
        classes = df.loc[(df.measurement_type == "Multiclass Confusion Matrix Class") & (df.uuid == uuid)].value.values

        con_mat = []

        for i in range(len(classes)):
            con_mat_row = []
            for j in range(len(classes)):
                con_mat_row.append(int(values[j + i * len(classes)]))
            con_mat.append(con_mat_row)

        matrix = pd.DataFrame(con_mat, index=classes, columns=classes)
        plt.figure(figsize=(12, 8))
        plt.title("Confusion Matrix for run " + uuid)
        sn.heatmap(matrix, annot=True)
        plt.show()


def plot_barh(df, measurement_type, title, xlabel):

    filtered_df = df.loc[df.measurement_type == measurement_type]
    uuids = filtered_df.uuid.unique()
    descriptions = filtered_df.description.unique()

    dic = {"uuids": uuids}
    for description in descriptions:
        dic[description] = filtered_df.loc[filtered_df.description == description].value.values.astype(np.float)

    df = pd.DataFrame(
        dic,
        index=uuids
    )

    ax = df.plot.barh(stacked=False)
    plt.title(title)
    plt.xlabel(xlabel)

    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.2f}".format(b.x1 - b.x0)
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

    plt.show()


def plot_time(df):

    plot_barh(df, "Time", "Time spent in phases", "Time in seconds")


def plot_throughput(df):

    plot_barh(df, "Throughput", "Throughput", "Seconds per entry")


def plot_latency(df):

    plot_barh(df, "Latency", "Latency", "Entries per second")


metrics_dict = {"Time": plot_time,
                "TTA": plot_TTA,
                "Memory": plot_memory,
                "Energy": plot_energy,
                "Multiclass Confusion Matrix": plot_confusion_matrix,
                "Latency": plot_latency,
                "Throughput": plot_throughput}


if __name__ == "__main__":
    visualize(["c147975f-8a5e-4a90-8f01-9bc62e63f33b", "c01850e8-8ab7-4819-9074-f2ea132317b4"], "so2sat_benchmark.db")
