import argparse
from datamodel import BenchmarkMetadata
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc
import os

import visualization

parser = argparse.ArgumentParser(description="Visualization CLI for End to End ML System Benchmark")

parser.add_argument("database", help="Sqlite Database file created by the benchmark package")
parser.add_argument("-u", "--uuid", help="UUID of the benchmark to visualize")

args = parser.parse_args()


def main():
    if not os.path.isfile(args.database):
        print(f"{args.database} not found.")
        return

    if not args.uuid:
        engine = create_engine(f'sqlite+pysqlite:///{args.database}')
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            q = session.query(BenchmarkMetadata.description,
                              BenchmarkMetadata.uuid,
                              BenchmarkMetadata.start_time).order_by(asc(BenchmarkMetadata.start_time))

            results = q.all()
            headers = ("Description", "UUID", "Timestamp")
            results = [tuple(map(str, result)) for result in results]
            for row in [headers] + results:
                print("{:<36}\t{:<36}\t{:<20}".format(*row))

            session.commit()

        finally:
            session.close()

    else:
        visualization.visualize(args.uuid, args.database)


if __name__ == "__main__":
    main()

