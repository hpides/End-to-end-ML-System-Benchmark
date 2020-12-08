import time
from datetime import datetime


class Measure(object):
    name = None

    def __init__(self, output_file):
        self.output_file = output_file

    def log(self, function_name, value):
        with open(self.output_file, 'a+') as out_file:
            out_file.write(f"{str(datetime.now())} --- {function_name} --- {self.name}: {value}\n")


class MeasureTime(Measure):
    name = "Time measurement"

    def __call__(self, func):
        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            self.log(func.__name__, time_taken)
            return result
        return inner
