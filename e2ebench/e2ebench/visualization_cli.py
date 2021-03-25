import argparse
import os

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from e2ebench.datamodel import Measurement
from e2ebench import metrics

parser = argparse.ArgumentParser(description="Visualization CLI for End to End ML System Benchmark")

parser.add_argument("database", help="Sqlite Database file created by the benchmark package")
parser.add_argument("-u", "--uuid", help="UUID of the benchmark to visualize", required=True)
parser.add_argument("-t", "--type", help="measurement type", required=True)
parser.add_argument("-d", "--desc", help="description", required=True)

args = parser.parse_args()

def main():

    engine = create_engine(f'sqlite+pysqlite:///{args.database}')
    Session = sessionmaker(bind=engine)
    session = Session()

    query_results = session.query(Measurement).filter_by(
                                                benchmark_uuid=args.uuid,
                                                measurement_type=args.type,
                                                description=args.desc).all()
                    
    if len(query_results) == 0:
        raise Exception("No entries found in database with given uuid, type and description")

    if len(query_results) > 1:
        raise Exception("Given combination of uuid, type and description is not unique")

    query_result = query_results.pop()
    Measurement_Class = metrics.measurement_type_mapper[query_result.measurement_type]
    measurement = Measurement_Class._from_serialized(query_result.value)
    measurement.visualize()

if __name__ == "__main__":
    main()
