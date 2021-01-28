from datamodel import Measurement, BenchmarkMetadata
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, desc, asc
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def visualize(uuid):
    engine = create_engine('sqlite+pysqlite:///backblaze_benchmark.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df_dict = make_dataframe_from_database(uuid, session)
        plot_measurement_type(df_dict, "Memory")

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
    plt.legend(loc=2)

    ax.yaxis.set_major_locator(ticker.LinearLocator(12))
    plt.show()


if __name__ == "__main__":
    visualize('8a989821-f2e9-48da-ab41-36cb0fc6f580')
