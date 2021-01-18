from decorators import MeasureTime, MeasureMemorySamples, MeasureLatency, MeasureThroughput
from config import parser as config
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datamodel import Base, Measurement
from uuid import uuid4

engine = create_engine('sqlite+pysqlite:///' + config['filepaths']['out_file'])
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

benchmark_uuid = str(uuid4())

# @MeasureTime(session)
# def print_this(string, times):
#     for i in range(times):
#         print(string)
#

@MeasureMemorySamples(session=session, benchmark_uuid=benchmark_uuid, interval=0.1)
@MeasureTime(session=session, benchmark_uuid=benchmark_uuid)
def bloat(minsize, maxsize, step):
    a = None
    for i in range(minsize, maxsize, step):
        a = np.random.rand(i, i)
    return a


def main():
    try:
        print(bloat(0, 2000, 100))
        query = session.query(Measurement)
        for row in query.all():
            print(row.__dict__)
    finally:
        session.close()


if __name__ == "__main__":
    main()
