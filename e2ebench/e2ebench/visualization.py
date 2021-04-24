from math import floor
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sn
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from e2ebench.datamodel import Measurement, BenchmarkMetadata

def visualize(uuids, database_file):
    engine = create_engine(f'sqlite+pysqlite:///{database_file}')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        values, meta = make_dataframe_from_database(uuids, session)
        metrics = values.measurement_type.unique()

        for metric in metrics:
            if metric == "Multiclass Confusion Matrix Class":           # helper class for MCCM, not a metric on its own
                continue
            for uuid in uuids:
                filtered_df = values.loc[(values.measurement_type == metric) & (values.uuid == uuid)]
                if len(filtered_df) != 0:
                    metrics_dict[metric](values, meta)
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

    meta_query = session.query(BenchmarkMetadata.uuid,
                               BenchmarkMetadata.description,
                               BenchmarkMetadata.start_time)

    filtered_measure_query = measure_query.filter_by(benchmark_uuid=uuids[0])
    filtered_meta_query = meta_query.filter_by(uuid=uuids[0])
    for uuid in range(1, len(uuids)):
        temp_measure_query = measure_query.filter_by(benchmark_uuid=uuids[uuid])
        temp_meta_query = meta_query.filter_by(uuid=uuids[uuid])
        filtered_measure_query = filtered_measure_query.union(temp_measure_query)
        filtered_meta_query = filtered_meta_query.union(temp_meta_query)
    filtered_measure_query = filtered_measure_query.order_by(asc(Measurement.datetime))
    filtered_meta_query = filtered_meta_query.order_by(asc(BenchmarkMetadata.start_time))

    values = pd.DataFrame(filtered_measure_query.all(), columns=["uuid", "datetime", "description", "measurement_type", "value", "unit"])
    meta = pd.DataFrame(filtered_meta_query.all(), columns=["uuid", "description", "start_time"])

    return values, meta


def plot_surface(values_df, meta_df):

    for uuid in values_df.uuid.unique():

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        start_time = meta_df.loc[(meta_df.uuid == uuid)].start_time.values[0]
        values = values_df.loc[(values_df.measurement_type == "Batch and Epoch")
                               & (values_df.uuid == uuid)].value.values.astype(float)
        if len(values) == 0:
            break

        epochs = np.arange(1.0, 11.0)
        batches = np.arange(1.0, 11.0)
        batches = np.power(2, batches)
        ax.set_yticks(np.log2(batches))
        ax.set_yticklabels(batches)
        ax.set_xticks(epochs)
        ax.set_xticklabels(epochs)
        epochs, batches = np.meshgrid(epochs, batches)
        values = np.reshape(values, (-1, 10))

        surf = ax.plot_surface(epochs, np.log2(batches), values, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("Batch size")
        ax.set_zlabel("loss")

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title("Batch size and epoch influence for run from " + str(start_time))
        plt.show()


def plot_time_based_graph(values_df, meta_df, measurement_type, title, xlabel, ylabel):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    filtered_df = values_df.loc[(values_df.measurement_type == measurement_type)]

    for uuid in filtered_df.uuid.unique():
        filtered_uuid_df = filtered_df.loc[(values_df.uuid == uuid)]
        start_time = meta_df.loc[(meta_df.uuid == uuid)].start_time.values[0]
        lastTimestamp, lastValue = 0, 0
        for description in values_df.description.unique():
            description_df = filtered_uuid_df.loc[values_df.description == description]
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
                        label=("Run from " + str(start_time)))

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.legend(loc=2)
    plt.title(title)

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


def plot_graph(values_df, meta_df, measurement_type, title, xlabel, ylabel, based_on):

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    df = values_df.loc[(values_df.measurement_type == measurement_type)]

    for uuid in df.uuid.unique():
        uuid_df = df.loc[(values_df.uuid == uuid)]
        start_time = meta_df.loc[(meta_df.uuid == uuid)].start_time.values[0]
        x_values = []
        for i in range(len(uuid_df.value.values)):
            if based_on == "Batches":
                x_values.append(2 ** (i + 1))
                ax.set_xscale('log')
            elif based_on == "Epochs":
                x_values.append(i + 1)
            else:
                x_values.append(10**floor(i/9)*((i % 9)+1)*0.00001)

        if not based_on == "LR":
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_values)
            ax.plot(x_values,
                    uuid_df.value.values.astype(float),
                    label=("Run from " + str(start_time)))
        else:
            plt.xticks(rotation=90)
            ax.plot(['{:.5f}'.format(x) for x in x_values],
                    uuid_df.value.values.astype(float),
                    label=("Run from " + str(start_time)))


    plt.legend(loc=2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.title(title)

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


def plot_barh(values_df, meta_df, measurement_type, title, xlabel):

    filtered_df = values_df.loc[values_df.measurement_type == measurement_type]
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


def plot_confusion_matrix(values_df, meta_df):

    for uuid in values_df.uuid.unique():

        start_time = meta_df.loc[(meta_df.uuid == uuid)].start_time.values[0]

        values = values_df.loc[(values_df.measurement_type == "Multiclass Confusion Matrix")
                               & (values_df.uuid == uuid)].value.values
        if len(values) == 0:
            break
        classes = values_df.loc[(values_df.measurement_type == "Multiclass Confusion Matrix Class")
                                & (values_df.uuid == uuid)].value.values

        con_mat = []

        for i in range(len(classes)):
            con_mat_row = []
            for j in range(len(classes)):
                con_mat_row.append(int(values[j + i * len(classes)]))
            con_mat.append(con_mat_row)

        matrix = pd.DataFrame(con_mat, index=classes, columns=classes)
        plt.figure(figsize=(12, 8))
        plt.title("Confusion Matrix for run at " + str(start_time))
        sn.heatmap(matrix, annot=True)
        plt.show()


def plot_memory(values, meta):
    plot_time_based_graph(values, meta, "Memory", "Memory usage", "Time in seconds", "MB used")


def plot_energy(values, meta):
    plot_time_based_graph(values, meta, "Energy", "Power consumption", "Time in seconds", "mJ of energy usage")


def plot_TTA(values, meta):
    plot_graph(values, meta, "TTA", "Time (Epochs) to Accuracy", "Total Epochs in training", "Accuracy", "Epochs")


def plot_loss(values, meta):
    plot_graph(values, meta, "Loss", "Training loss in each epoch", "Total Epochs in training", "Loss", "Epochs")


def plot_batch_influence(values, meta):
    plot_graph(values, meta, "Batch", "Training loss regarding batch size", "Batch size", "Loss", "Batches")


def plot_lr_influence(values, meta):
    plot_graph(values, meta, "Learning Rate", "Learning rate influence on loss", "Learning Rate", "Loss", "LR")


def plot_time(values, meta):
    plot_barh(values, meta, "Time", "Time spent in phases", "Time in seconds")


def plot_throughput(values, meta):
    plot_barh(values, meta, "Throughput", "Throughput", "Seconds per entry")

def plot_latency(values, meta):
    plot_barh(values, meta, "Latency", "Latency", "Entries per second")

def plot_hyperparameters(df_from_cli):
    color_scale = px.colors.diverging.Tealrose
    for row in df_from_cli.iterrows():
        deserialized = pickle.loads(row['bytes'])
        hyperparams = deserialized['hyperparameters']
        hyperparam_df = deserialized['df']
        target = deserialized['target']
        target_low_means_good = deserialized['low_means_good']

        if not low_means_good:
            color_scale = list(reversed(color_scale))

        fig = px.parallel_coordinates(hyperparam_df, 
                                      color=target,
                                      dimensions=hyperparams,
                                      color_continuous_scale=color_scale)
        fig.show()

def plot_confusion_matrix_plotly(df_from_cli):
    for _, row in df_from_cli.iterrows():
        deserialized = pickle.loads(row['bytes'])
        matrix = deserialized['matrix']
        labels = deserialized['labels']
        matrix_str = [[str(y) for y in x] for x in matrix]
        fig = ff.create_annotated_heatmap(matrix, 
                                        x=labels,
                                        y=labels,
                                        annotation_text=matrix_str,
                                        colorscale=px.colors.diverging.Tealrose
                                        )

        layout = {
            "xaxis" : {"title" : "Predicted Value"},
            "yaxis" : {"title" : "Real Value"},
        }

        fig.show()

def plot_throughput_matplotlib(df_from_cli):
    df_from_cli['measurement_time_str'] = df_from_cli['measurement_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_from_cli['x_labels'] = df_from_cli['measurement_time_str'] + " " + df_from_cli['measurement_desc']
    df_from_cli.rename(columns={'measurement_value' : 'throughput'}, inplace=True)
    df_from_cli.sort_values(by='measurement_time', inplace=True)
    
    ax = df_from_cli.plot.barh(x='x_labels', y='throughput', stacked=False)
    plt.title("Throughput")
    plt.xlabel("Seconds per entry")

    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        val = "{:.2f}".format(b.x1 - b.x0)
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

    plt.show()


def plot_throughput_plotly(df_from_cli):
    df_from_cli['measurement_time_str'] = df_from_cli['measurement_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_from_cli['x_labels'] = df_from_cli['measurement_time_str'] + " " + df_from_cli['measurement_desc']
    df_from_cli.rename(columns={'measurement_value' : 'throughput'}, inplace=True)
    df_from_cli.sort_values(by='measurement_time', inplace=True)
    fig = px.bar(df_from_cli, 
                 x='x_labels', y='throughput',
                 hover_data={'uuid': True,
                             'type': df_from_cli['measurement_type'],
                             'description': df_from_cli['measurement_desc'],
                             'meta description' : df_from_cli['meta_desc'].replace('', 'None'),
                             'meta start time' : df_from_cli['meta_start_time'].dt.strftime("%Y-%m-%d\t%H:%M:%S")},
                 color='throughput',
                 labels={'x_labels' : 'Measurement', 'throughput' : 'Throughput'}
    )
    fig.show()

visualization_func_mapper = {
                "Loss": plot_loss,
                "Batch": plot_batch_influence,
                "Batch and Epoch": plot_surface,
                "Learning Rate": plot_lr_influence,
                "time": plot_time,
                "TTA": plot_TTA,
                "memory": plot_memory,
                "energy": plot_energy,
                "Multiclass Confusion Matrix": plot_confusion_matrix,
                "Latency": plot_latency,
                "throughput": plot_throughput_matplotlib,
                "hyperparameters" : plot_hyperparameters,
                "confusion-matrix" : plot_confusion_matrix_plotly,
                "loss" : plot_loss,
                "tta" : plot_TTA}

