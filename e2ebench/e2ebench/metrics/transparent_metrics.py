import time
import resource
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import pyRAPL
from math import floor

class Measure(object):
    name = None

    def __init__(self, benchmark, description):
        self.benchmark = benchmark
        self.description = description

class MeasureTime(Measure):
    measurement_type = "Time"

    def __call__(self, func):
        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            self.benchmark.log(self.description, self.measurement_type, time_taken, "seconds")
            return result

        return inner

class MeasureMemorySamples(Measure):
    measurement_type = "Memory"

    def __init__(self, benchmark, description, interval=1.0):
        super().__init__(benchmark, description)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        def inner(*args, **kwargs):
            with ThreadPoolExecutor() as tpe:
                try:
                    func_thread = tpe.submit(func, *args, **kwargs)
                    while not func_thread.done() and self.keep_measuring:
                        self.log_memory()
                    result = func_thread.result()
                finally:
                    self.keep_measuring = False
                return result
        return inner

    def log_memory(self):
        measurement_value = f"{round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}"
        self.benchmark.log(self.description, self.measurement_type, measurement_value, "MB")
        time.sleep(self.interval)
        return measurement_value


class MeasureEnergy(Measure):
    measurement_type = "Energy"

    def __init__(self, benchmark, description, interval=1.0):
        super().__init__(benchmark, description)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        def inner(*args, **kwargs):
            pyRAPL.setup()
            self.meter = pyRAPL.Measurement('bar')
            with ThreadPoolExecutor() as tpe:
                try:
                    tpe.submit(self.log_energy)
                    result = func(*args, **kwargs)
                finally:
                    self.keep_measuring = False
                    self.meter.end()
                return result
        return inner

    def log_energy(self):
        while self.keep_measuring:
            measurement_value = self.meter.result.pkg[0]
            self.benchmark.log(self.description, self.measurement_type, measurement_value/1000, "mJ")
            self.meter.begin()
            time.sleep(self.interval)
   

class MeasureMemoryTracemalloc(Measure):
    measurement_type = "Memory (tracemalloc)"

    def __init__(self, benchmark, description, interval=1.0):
        super().__init__(benchmark, description)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        def inner(*args, **kwargs):
            with ThreadPoolExecutor() as tpe:
                try:
                    if not tracemalloc.is_tracing():
                        tracemalloc.start()
                    func_thread = tpe.submit(func, *args, **kwargs)
                    while not func_thread.done() and self.keep_measuring:
                        self.log_memory()
                    result = func_thread.result()
                finally:
                    self.keep_measuring = False
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                return result
        return inner

    def log_memory(self):
        measurement_value = tracemalloc.get_traced_memory()[0] / (2**20)
        self.benchmark.log(self.description, self.measurement_type, measurement_value, "MiB")
        time.sleep(self.interval)


class MeasureMemoryPsutil(Measure):
    measurement_type = "Memory (psutil)"

    def __init__(self, benchmark, description, interval=1.0):
        super().__init__(benchmark, description)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        def inner(*args, **kwargs):
            self.pid = os.getpid()
            self.process = psutil.Process(self.pid)
            with ThreadPoolExecutor() as tpe:
                try:
                    tpe.submit(self.log_memory)
                    result = func(*args, **kwargs)
                finally:
                    self.keep_measuring = False
                return result
        return inner

    def log_memory(self):
        while self.keep_measuring:
            measurement_value = self.process.memory_info()[0] / (2**20)
            self.benchmark.log(self.description, self.measurement_type, measurement_value, "MiB")
            time.sleep(self.interval)