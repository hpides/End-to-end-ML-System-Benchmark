from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from .datamodel import Base, Measurement, BenchmarkMetadata
from queue import Queue
from threading import Thread, Event
from time import sleep


class Benchmark:
    def __init__(self, db_file, description=""):
        self.db_file = db_file
        self.close_event = Event()
        self.uuid = str(uuid4())
        self.description = description
        self.queue = Queue()

        t = Thread(target=self.__database_thread_func)
        t.start()

    def close(self):
        self.close_event.set()

    def __database_thread_func(self):
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
        measurement = Measurement(datetime=datetime.now(),
                                  benchmark_uuid=self.uuid,
                                  description=description,
                                  measurement_type=measure_type,
                                  value=value,
                                  unit=unit)
        self.queue.put(measurement)

