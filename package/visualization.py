from .datamodel import Measurement, BenchmarkMetadata
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc, asc

import pandas as pd
# import seaborn as sn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import plot_confusion_matrix


def visualize(uuid):
    engine = create_engine('sqlite+pysqlite:///backblaze_benchmark.db')
    # engine = create_engine('sqlite+pysqlite:///so2sat_benchmark.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df_dict = make_dataframe_from_database(uuid, session)
        plot_measurement_type(df_dict, "Memory")
        plot_confusion_matrix(df_dict)

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
    ax.set_ylim(bottom=0)
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
            con_mat_row.append(int(values[j+i*len(classes)]))
        con_mat.append(con_mat_row)

    matrix = pd.DataFrame(con_mat, index=classes, columns=classes)
    plt.figure(figsize=(12, 8))
    plt.title("Confusion Matrix")
    sn.heatmap(matrix, annot=True)
    plt.show()


if __name__ == "__main__":
    visualize('8a989821-f2e9-48da-ab41-36cb0fc6f580')
    # visualize("a8c2115c-0d6a-4cbc-ad47-445289d136fc")
