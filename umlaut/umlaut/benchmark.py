from datetime import datetime
import os
import pickle
from queue import Queue
from threading import Thread, Event
from time import sleep
from uuid import uuid4

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .datamodel import Base, Measurement, BenchmarkMetadata

class Benchmark:
    """A class that manages the database entries for the measured metrics which are logged into the database.

    Parameters
    ----------
    db_file : str
        The path of the database file
    description : str, optional
        The description of the whole pipeline use case. Even though the description is optional, it should be set
        so the database entries are distinguishable without evaluating the uuid's.
        This parameter is ignored for Benchmark objects initialized in mode 'r'.
    mode : str, default='a'
        One of ['w', 'a', 'r']. The mode corresponds to conventional python file handling modes.
        Modes 'a' and 'w' are used for storing metrics in a database during a pipeline run
        and 'r' is used for querying metrics from the database.

    Attributes
    ----------
    db_file : str
        path to the database file
    mode : str
        mode of the Benchmark object. One of ['w', 'a', 'r'].
    description : str
        description of the pipeline run. Not relevant if mode is 'r'.
    session : sqlalchemy.orm.session.Session
        SQLalchemy session
    """

    def __init__(self, db_file, description="", mode="a"):
        self.db_file = db_file
        self.description = description
        self.mode = mode

        if mode == 'r':
            if not os.path.exists(self.db_file):
                raise FileNotFoundError("Cannot open a non-existing file in reading mode.")
            engine = create_engine('sqlite+pysqlite:///' + self.db_file)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.session = Session()

        if mode == 'w':
            if os.path.exists(self.db_file):
                os.remove(self.db_file)
        
        if mode in ['w', 'a']:
            self.close_event = Event()
            self.uuid = str(uuid4())
            self.queue = Queue()

            self._db_thread = Thread(target=self._database_thread_func)
            self._db_thread.start()

    def query(self, *args, **kwargs):
        """
        Send queries to the database file.
        You can send queries in the same manner you would query an SQLalchemy session.

        This method only works in mode 'r'.
        """
        if self.mode != "r":
            raise Exception("Invalid file mode. Mode must be \"r\" to send queries.")

        return self.session.query(*args, **kwargs)

    def close(self):
        """
        Close the Benchmark object.
        For Benchmark objects used in mode 'r', the SQLalchemy session is closed.
        For the remaining modes the session is closed and all collected metrics are written to the database file.        
        """
        if self.mode == 'r':
            self.session.close()
        else:
            self.close_event.set()
            self._db_thread.join()

    def _database_thread_func(self):
        """The function that manages the threading."""
        engine = create_engine('sqlite+pysqlite:///' + self.db_file)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        session.add(BenchmarkMetadata(uuid=self.uuid,
                                      meta_description=self.description,
                                      meta_start_time=datetime.now()))
        session.commit()

        try:
            while True:
                log_staged = False
                while not self.queue.empty():
                    sleep(0)
                    measurement = self.queue.get()
                    #measurement.measurement_data = str(measurement.measurement_data)
                    session.add(measurement)
                    log_staged = True
                if log_staged:
                    session.commit()
                if self.close_event.isSet() and self.queue.empty():
                    break
                sleep(0)
        finally:
            session.close()

    def log(self, description, measure_type, value, unit='', method_name=""):
        measurement = Measurement(measurement_datetime=datetime.now(),
                                  uuid=self.uuid,
                                  measurement_description=description,
                                  measurement_type=measure_type,
                                  measurement_data=value,
                                  measurement_unit=unit,
                                  measured_method_name=method_name)
        self.queue.put(measurement)

class VisualizationBenchmark(Benchmark):
    def __init__(self, db_file):
        super().__init__(db_file, mode='r')

    def query_all_meta(self):
        """
        Returns a dataframe of all entries in BenchmarkMetadata and sets uuid as the index.
        """
        query = self.query(BenchmarkMetadata.uuid,
                           BenchmarkMetadata.meta_description,
                           BenchmarkMetadata.meta_start_time)
        col_names = [col_desc['name'] for col_desc in query.column_descriptions]

        return pd.DataFrame(query.all(), columns=col_names).set_index('uuid')

    def query_all_uuid_type_desc(self):
        """
        Returns a dataframe of all entries in Measurement and sets Measurement.id as the index.
        """      
        query = self.query(Measurement.id,
                           Measurement.uuid,
                           Measurement.measurement_type,
                           Measurement.measurement_description)
        col_names = [col_desc['name'] for col_desc in query.column_descriptions]

        return pd.DataFrame(query.all(), columns=col_names).set_index('id')

    def join_visualization_queries(self, uuid_type_desc_df):
        """
        Joins all remaining database columns from both tables to uuid_type_desc_df.

        Parameters
        ----------
        uuid_type_desc_df : pandas.DataFrame
            Dataframe containing the columns uuid, measurement_type, measurement_description
            and Measurement.id as the index
            (same schema as returned by VisualizationBenchmark.query_all_uuid_type_desc()).

        Returns
        -------
        joined_df : pandas.DataFrame
            uuid_type_desc_df joined with the remaining database columns from both tables.
            Measurement.id is still the index in joined_df.
        """
        meta_query = self.query(BenchmarkMetadata.uuid,
                                BenchmarkMetadata.meta_start_time,
                                BenchmarkMetadata.meta_description).filter(
                                    BenchmarkMetadata.uuid.in_(uuid_type_desc_df['uuid']))
        meta_col_names = [col_desc['name'] for col_desc in meta_query.column_descriptions]
        meta_df = pd.DataFrame(meta_query.all(), columns=meta_col_names)
        print("META_COLUMN_NAMES", meta_col_names)

        measurement_query = self.query(Measurement.id,
                                       Measurement.measurement_datetime,
                                       Measurement.measurement_data,
                                       Measurement.measurement_unit,
                                       Measurement.measured_method_name).filter(
                                            Measurement.id.in_(uuid_type_desc_df.index))
        measure_col_names = [col_desc['name'] for col_desc in measurement_query.column_descriptions]
        print("COLUMN_NAMES", measure_col_names)
        measurement_df = pd.DataFrame(measurement_query.all(), columns=measure_col_names)
        #measurement_df['measurement_data'] = measurement_df['measurement_data'].map(pickle.loads)
        measurement_df['measurement_data'] = measurement_df['measurement_data']

        joined_df = uuid_type_desc_df.reset_index().merge(meta_df, on='uuid')
        joined_df = joined_df.merge(measurement_df, on='id')

        return joined_df.set_index('id')
