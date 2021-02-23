from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from .datamodel import Base, Measurement, BenchmarkMetadata


class Benchmark:
    def __init__(self, db_file, description=""):
        self.db_file = db_file
        engine = create_engine('sqlite+pysqlite:///' + self.db_file)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.uuid = str(uuid4())

        self.session.add(BenchmarkMetadata(uuid=self.uuid, description=description, start_time=datetime.now()))
        self.session.commit()
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def log(self, description, measure_type, value, unit=''):
        measurement = Measurement(datetime=datetime.now(),
                                  benchmark_uuid=self.uuid,
                                  description=description,
                                  measurement_type=measure_type,
                                  value=value,
                                  unit=unit)
        self.session.add(measurement)
        self.session.commit()

    def close(self):
        self.session.close()
