from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import pickle
import psutil
import threading
import time
import pyRAPL


class BenchmarkSupervisor:
    def __init__(self, metrics, benchmark):
        self.metrics = sorted(metrics)
        self.benchmark = benchmark

    def __call__(self, func):
        def inner(*args, **kwargs):
            finish_event = threading.Event()
            with ThreadPoolExecutor(max_workers=len(self.metrics)) as tpe:
                try:
                    self.__before()
                    threads = self.__meanwhile(tpe, finish_event)
                    function_result = func(*args, **kwargs)
                    finish_event.set()
                    for thread in threads:
                        thread.result()
                    self.__after()
                    self.__log()
                finally:
                    finish_event.set()

            return function_result

        return inner

    def __before(self):
        for metric in reversed(self.metrics):
            metric.before()

    def __after(self):
        for metric in self.metrics:
            metric.after()

    def __meanwhile(self, tpe, finish_event):
        threads = []
        for metric in self.metrics:
            if metric.needs_threading:
                threads.append(tpe.submit(metric.meanwhile, finish_event))
        return threads

    def __log(self):
        for metric in self.metrics:
            metric.log(self.benchmark)


class Metric:
    priority = 0
    needs_threading = False

    def __init__(self, description):
        self.description = description

    def __lt__(self, other):
        return self.priority < other.priority

    def before(self):
        pass

    def after(self):
        pass

    def meanwhile(self, finish_event):
        pass

    def serialize(self):
        return pickle.dumps(self.data)

    def log(self, benchmark):
        pass


class TimeMetric(Metric):
    priority = 0
    measure_type = 'time'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = after_time - self.before_time

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='s')


class MemoryMetric(Metric):
    priority = 1
    measure_type = 'memory'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        process = psutil.Process(os.getpid())
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            self.measurements.append(process.memory_info()[0] / (2 ** 20))
            time.sleep(self.interval)

    def after(self):
        self.data = {
            'timestamps' : self.timestamps,
            'measurements' : self.measurements
        }

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize())


class PowerMetric(Metric):
    priority = 1
    measure_type = 'power'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval
        self.meter = None

    def before(self):
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')
        self.data = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.meter.begin()
            time.sleep(self.interval)
            self.meter.end()
            power = sum(map(lambda x: x / self.meter.result.duration, self.meter.result.pkg))
            self.data.append(power)

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize())