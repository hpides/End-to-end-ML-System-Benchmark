from PyInquirer import prompt

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc

from e2ebench.datamodel import Measurement
from e2ebench import metrics

def main():
    question0 = [
        {
            "type": "input",
            "name": "database",
            "message": "Please choose a database file."
        }
    ]

    db_file = prompt(question0).get("database")
    
    engine = create_engine(f'sqlite+pysqlite:///{db_file}')
    Session = sessionmaker(bind=engine)
    session = Session()

    uuid_list_raw = session.query(Measurement.benchmark_uuid).all()
    uuid_list = list(set([r[0] for r in uuid_list_raw]))

    question1 = [
        {
            "type": "list",
            "name": "uuid",
            "message": "Please choose a UUID.",
            "choices": uuid_list
        }
    ]

    uuid = prompt(question1).get("uuid")
    type_list_raw = session.query(Measurement.measurement_type).filter_by(benchmark_uuid=uuid).all()
    type_list = list(set([r[0] for r in type_list_raw]))


    question2 = [
        {
            "type": "list",
            "name": "measurement_type",
            "message": "Please choose one of the following types.",
            "choices": type_list
        }
    ]

    measurement_type = prompt(question2).get("measurement_type")
    desc_list_raw = session.query(Measurement.description).filter_by(
        benchmark_uuid=uuid,
        measurement_type=measurement_type).all()
    desc_list = list(set([r[0] for r in desc_list_raw]))

    question3 = [
        {
            "type": "list",
            "name": "measurement_description",
            "message": "Please choose one of the following descriptions.",
            "choices": desc_list
        }
    ]

    measurement_description = prompt(question3).get("measurement_description")
    query_results = session.query(Measurement).filter_by(
        benchmark_uuid=uuid,
        measurement_type=measurement_type,
        description=measurement_description).all()

    if len(query_results) == 0:
        raise Exception("No entries found in database with given uuid, type and description")

    if len(query_results) > 1:
        raise Exception("Given combination of uuid, type and description is not unique")

    query_result = query_results.pop()
    Visualizer_Class = metrics.measurement_type_mapper[query_result.measurement_type]
    measurement = Visualizer_Class(query_result.value)
    measurement.visualize()
    session.close()

if __name__ == "__main__":
    main()