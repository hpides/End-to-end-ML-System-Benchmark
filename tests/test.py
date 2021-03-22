import unittest
import e2ebench
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, asc
from e2ebench.e2ebench.datamodel import Measurement
import pandas as pd
import time
import os
import numpy as np

bm = e2ebench.Benchmark('test_benchmark.db', description="Database for testing the benchmark package")


def mock_function():
    result = 1
    for num in range(1, 11):
        result *= num
    return {"loss": [0.5, 0.3, 0.2, 0.01], "accuracy": [60.0, 75.5, 88.0, 95.0],
            "confusion matrix": [[3.5,32.5],[2.4,25.3]],
            "classes": ["class1", "class2", "class3", "class4"], "num_entries": 20}


def get_database_df(measurement):
    engine = create_engine(f'sqlite+pysqlite:///test_benchmark.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    measure_query = session.query(Measurement.benchmark_uuid,
                                  Measurement.datetime,
                                  Measurement.description,
                                  Measurement.measurement_type,
                                  Measurement.value,
                                  Measurement.unit)

    #measure_query = measure_query.filter_by(measurement_type=measurement)
    values = pd.DataFrame(measure_query.all())
    print(values)
    return values


class TestDecorators(unittest.TestCase):

    def setUp(self):
        if os.path.exists("test_benchmark.db"):
            os.remove("test_benchmark.db")
        #else:
        #    print("The file does not exist")

    def tearDown(self):
        if os.path.exists("test_benchmark.db"):
            os.remove("test_benchmark.db")
        #else:
        #    print("The file does not exist")

    def test_time(self):
        @e2ebench.MeasureTime(bm, description="Time")
        def dec_mock_function():
            mock_function()
        dec_mock_function()
        bm.close()
        time.sleep(2)
        df = get_database_df("Time")
        self.assertEqual(len(df), 1)

    def test_confusion_matrix(self):
        @e2ebench.MeasureMulticlassConfusion(bm, description="Multiclass Confusion Matrix")
        def dec_mock_function():
            return mock_function()
        result = dec_mock_function()
        print("conf:", result["confusion matrix"][5])
        bm.close()
        time.sleep(2)
        df = get_database_df("Multiclass Confusion Matrix")
        self.assertEqual(len(df), 1)

    def test_loss(self):
        @e2ebench.MeasureLoss(bm, description="Loss")
        def dec_mock_function():
            return mock_function()
        result = dec_mock_function()
        bm.close()
        time.sleep(2)
        df = get_database_df("Loss")
        self.assertEqual(len(df), 1)

if __name__ == '__main__':
    unittest.main()
