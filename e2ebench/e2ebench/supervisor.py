import threading
import time
from concurrent.futures import ThreadPoolExecutor
import os
import psutil
from datetime import datetime


class BenchmarkSupervisor:
    def __init__(self, metrics, description):
        self.metrics = sorted(metrics)
        self.description = description

    def __call__(self, func):
        def inner(*args, **kwargs):
            finish_event = threading.Event()
            with ThreadPoolExecutor() as tpe:
                try:
                    self.__before()
                    threads = self.__meanwhile(tpe, finish_event)
                    function_result = func(*args, **kwargs)
                    finish_event.set()
                    self.__after()
                    for thread in threads:
                        thread.result(timeout=None)
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
            threads.append(tpe.submit(metric.meanwhile, finish_event))
        return threads

    def __log(self):
        for metric in self.metrics:
            print(metric.data())


class Metric:
    priority = 0

    def __lt__(self, other):
        return self.priority < other.priority

    def before(self):
        pass

    def after(self):
        pass

    def meanwhile(self, finish_event):
        pass


class TimeMetric(Metric):
    priority = 0

    def __init__(self):
        self.before_time = None
        self.after_time = None

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        self.after_time = time.perf_counter()

    def data(self):
        return self.after_time-self.before_time


class MemoryMetric(Metric):
    priority = 1

    def __init__(self, interval=1):
        self.interval = interval
        self.measurements = []

    def meanwhile(self, finish_event):
        process = psutil.Process(os.getpid())
        while not finish_event.isSet():
            measurement_value = process.memory_info()[0] / (2 ** 20)
            self.measurements.append((datetime.now(), measurement_value))

    def data(self):
        return self.measurements

