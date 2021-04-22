import argparse
import os

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from e2ebench.datamodel import Measurement
from e2ebench.datamodel import BenchmarkMetadata
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

    uuids = args.uuid.replace(" ","").split(",")
    print(uuids)

    measure_query = session.query(Measurement.benchmark_uuid,
                                  Measurement.datetime,
                                  Measurement.description,
                                  Measurement.measurement_type,
                                  Measurement.value,
                                  Measurement.unit)

    meta_query = session.query(BenchmarkMetadata.uuid,
                               BenchmarkMetadata.description,
                               BenchmarkMetadata.start_time)

    filtered_measure_query = measure_query.filter_by(benchmark_uuid=uuids[0],
                                                     measurement_type=args.type,
                                                     description=args.desc)
    filtered_meta_query = meta_query.filter_by(uuid=uuids[0])
    for uuid in range(1, len(uuids)):
        temp_measure_query = measure_query.filter_by(benchmark_uuid=uuids[uuid],
                                                     measurement_type=args.type,
                                                     description=args.desc)
        temp_meta_query = meta_query.filter_by(uuid=uuids[uuid])
        filtered_measure_query = filtered_measure_query.union(temp_measure_query)
        filtered_meta_query = filtered_meta_query.union(temp_meta_query)
                    
    if len(filtered_measure_query.all()) == 0:
        raise Exception("No entries found in database with given uuid, type and description")

    values = []
    for result in filtered_measure_query.all():
        values.append(result.value)

    starts = []
    for result in filtered_meta_query.all():
        starts.append(result.start_time)

    filtered_measure_query = filtered_measure_query.all().pop()

    Visualizer_Class = metrics.measurement_type_mapper[args.type]
    measurement = Visualizer_Class(values)
    measurement.visualize(uuids, args.desc, starts)

if __name__ == "__main__":
    main()
