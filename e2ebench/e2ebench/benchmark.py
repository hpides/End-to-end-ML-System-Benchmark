from datetime import datetime
import os
from queue import Queue
from threading import Thread, Event
from time import sleep
from uuid import uuid4
from numpy.testing._private.utils import measure

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .datamodel import Base, Measurement, BenchmarkMetadata


class Benchmark:
    """A class, that manages the database entries for the measured metrics which are logged into the database.

    Attributes
    ----------
    db_file : str
        The database file where the metrics should be stored. The mode of db_file is 'append'.
    description : str, optional
        The description of the whole pipeline use case. Even though the description is optional, it should be set
        so the database entries are distinguishable without evaluating the uuid's.
    description : str
        The description of the metric.
    measure_type : str
        The measurement type of the metric.
    value : :obj:'bytes'
        The bytes object of the data which should be logged.
    unit : str
        The unit of the measured values.

    Methods
    -------
    close
        The function that sets the close event and joins the results of the threads.
    __database_thread_func
        The function that manages the threading.
    log(description, measure_type, value, unit='')
        Logging of measured metrics into the database.
    """
    def __init__(self, db_file, description="", mode="a"):
        """ Initialisation of the benchmark object.

        Parameters
        ----------
        db_file : str
            The database file where the metrics should be stored. The mode of db_file is 'append'.
        description : str, optional
            The description of the whole pipeline use case. Even though the description is optional, it should be set
            so the database entries are distinguishable without evaluating the uuid's.
        """

        self.db_file = db_file
        self.description = description
        self.mode = mode

        if mode == 'r':
            if not os.path.isfile(self.db_file):
                raise FileNotFoundError("Cannot open a non-existing file in reading mode.")
            engine = create_engine('sqlite+pysqlite:///' + self.db_file)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.session = Session()

        if mode == 'w':
            if os.path.isfile(self.db_file):
                os.remove(self.db_file)
        
        if mode in ['w', 'a']:
            self.close_event = Event()
            self.uuid = str(uuid4())
            self.queue = Queue()

            self.__db_thread = Thread(target=self.__database_thread_func)
            self.__db_thread.start()
        

    def query(self, entities, filters=[], return_pandas=True):
        if self.mode != "r":
            raise Exception("Invalid file mode. Mode must be \"r\" to send queries.")

        query = self.session.query(*entities).filter(*filters)
        col_names = [col['name'] for col in query.column_descriptions]
        query_result = query.all()

        if return_pandas:
            query_result = pd.DataFrame(query_result, columns = col_names)

        return query_result

    def query_all_uuid_type_desc(self):       
        df =  self.query([Measurement.id,
                          Measurement.uuid,
                          Measurement.measurement_type,
                          Measurement.measurement_description])
        df.set_index('id', inplace=True)

        return df

    def join_visualization_queries(self, uuid_type_desc_df):
        meta_df = self.query([BenchmarkMetadata.uuid,
                              BenchmarkMetadata.meta_start_time,
                              BenchmarkMetadata.meta_description],
                              filters=[BenchmarkMetadata.uuid.in_(uuid_type_desc_df['uuid'])])

        measurement_df = self.query([Measurement.id,
                                     Measurement.measurement_datetime,
                                     Measurement.measurement_data,
                                     Measurement.measurement_unit],
                                    filters=[Measurement.id.in_(uuid_type_desc_df.index)])

        df = uuid_type_desc_df.reset_index().merge(meta_df, on='uuid')
        df = df.merge(measurement_df, on='id')

        return df.set_index('id')

    def close(self):
        """The function that sets the close event and joins the results of the threads."""
        if self.mode == 'r':
            self.session.close()
        else:
            self.close_event.set()
            self.__db_thread.join()

    def __database_thread_func(self):
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
                    session.add(measurement)
                    log_staged = True
                if log_staged:
                    session.commit()
                if self.close_event.isSet() and self.queue.empty():
                    break
                sleep(0)
        finally:
            session.close()

    def log(self, description, measure_type, value, unit=''):
        """Logging of measured metrics into the database.

        Parameters
        ----------
        description : str
            The description of the metric.
        measure_type : str
            The measurement type of the metric.
        value : :obj:'bytes'
            The bytes object of the data which should be logged.
        unit : str
            The unit of the measured values.

        Returns
        -------
        Measurement
            Measurement object with updated datetime, benchmark_uuid, description, measurement_type, value and unit.
        """
        measurement = Measurement(measurement_datetime=datetime.now(),
                                  uuid=self.uuid,
                                  measurement_description=description,
                                  measurement_type=measure_type,
                                  measurement_data=value,
                                  measurement_unit=unit)
        self.queue.put(measurement)

