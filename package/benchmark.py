from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from uuid import uuid4

from .datamodel import Base, Measurement

class Benchmark:
    def __init__(self, db_file):
        self.db_file = db_file
        engine = create_engine('sqlite+pysqlite:///' + self.db_file)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.uuid = str(uuid4())
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        

    def log(self, func_name, measure_type, value):
        # with open(self.output_file, 'a+') as out_file:
        #     out_file.write(f"{str(datetime.now())} --- {self.func_name} --- {self.name}: {value}\n")
        measurement = Measurement(datetime=datetime.now(),
                                  benchmark_uuid=self.uuid,
                                  function_name=func_name,
                                  measurement_type=measure_type,
                                  value=value)
        self.session.add(measurement)
        self.session.commit()

    def close(self):
        self.session.close()