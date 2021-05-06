from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from .datamodel import Base, Measurement, BenchmarkMetadata
from queue import Queue
from threading import Thread, Event
from time import sleep


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
    def __init__(self, db_file, description=""):
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
        self.close_event = Event()
        self.uuid = str(uuid4())
        self.description = description
        self.queue = Queue()

        self.__db_thread = Thread(target=self.__database_thread_func)
        self.__db_thread.start()

    def close(self):
        """The function that sets the close event and joins the results of the threads."""
        self.close_event.set()
        self.__db_thread.join()

    def __database_thread_func(self):
        """The function that manages the threading."""
        engine = create_engine('sqlite+pysqlite:///' + self.db_file)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        session.add(BenchmarkMetadata(uuid=self.uuid, description=self.description, start_time=datetime.now()))
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
        measurement = Measurement(datetime=datetime.now(),
                                  benchmark_uuid=self.uuid,
                                  description=description,
                                  measurement_type=measure_type,
                                  value=value,
                                  unit=unit)
        self.queue.put(measurement)

