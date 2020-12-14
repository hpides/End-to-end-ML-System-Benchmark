import time
import resource
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class Measure(object):
    name = None

    def __init__(self, output_file):
        self.output_file = output_file
        self.func_name = None

    def log(self, value):
        with open(self.output_file, 'a+') as out_file:
            out_file.write(f"{str(datetime.now())} --- {self.func_name} --- {self.name}: {value}\n")

    def __call__(self, func):
        self.func_name = func.__name__


class MeasureTime(Measure):
    name = "Time measurement"

    def __call__(self, func):
        super().__call__(func)

        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            self.log(time_taken)
            return result
        return inner


class MeasureMemorySamples(Measure):
    name = "Memory measurement by sampling"

    def __init__(self, output_file, interval):
        super().__init__(output_file)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        super().__call__(func)

        def inner(*args, **kwargs):
            inner.__name__ = self.func_name
            with ThreadPoolExecutor() as tpe:
                tpe.submit(self.measure_memory)
                try:
                    func_thread = tpe.submit(func, *args, **kwargs)
                    result = func_thread.result()
                finally:
                    self.keep_measuring = False
                return result
        return inner

    def measure_memory(self):
        while self.keep_measuring:
            self.log(f"{round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)} MB")
            time.sleep(self.interval)
