from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import os
import pickle
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import psutil
import pyRAPL


class BenchmarkSupervisor:
    """A supervisor object managing all supervised metrics

    This object should be used as a decorator.

    Parameters
    ----------
    metrics: list of Metric
        A list of metrics to be collected while running the decorated function
    benchmark: Benchmark
        The central benchmark object used in the pipeline
    """
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
    """The Metric object from which all supervised metric objects inherit"""
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
    """The metric object to measure the time taken for the execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    """
    priority = 0
    measure_type = 'time'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = after_time - self.before_time

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='sec')


class MemoryMetric(Metric):
    """The metric object to measure memory used in the execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between memory measurements
    """
    priority = 3
    measure_type = 'memory'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        self.process = psutil.Process(os.getpid())
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            self.measurements.append(self.process.memory_info().rss / (2 ** 20))
            time.sleep(self.interval)

    def after(self):
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit="MiB")

        
class EnergyMetric(Metric):
    """The metric object to measure energy used in the execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    """
    priority = 1
    measure_type = 'energy'
    needs_threading = False

    def before(self):
        try:
            pyRAPL.setup()
            self.meter = pyRAPL.Measurement('bar')
            self.meter.begin()
            self.successful = True
        except FileNotFoundError:
            logging.warning("RAPL file not found. Perhaps you are using a platform that does not support RAPL (for example Windows)")
            self.successful = False
        except PermissionError:
            logging.warning("PermissionError occured while reading RAPL file. Fix with \"sudo chmod -R a+r /sys/class/powercap/intel-rapl\"")
            self.successful = False

    def after(self):
        if self.successful:
            self.meter.end()
            self.data = sum(self.meter.result.pkg)

    def log(self, benchmark):
        if self.successful:
            benchmark.log(self.description, self.measure_type, self.serialize(), unit='µJ')


class PowerMetric(Metric):
    """The metric object to measure power used in the execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between memory measurements
    """
    priority = 3
    measure_type = 'power'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval
        self.meter = None

    def before(self):
        try:
            pyRAPL.setup()
            self.meter = pyRAPL.Measurement('bar')
            self.measurements = []
            self.timestamps = []
            self.meter.begin()
            self.successful = True
        except FileNotFoundError:
            logging.debug("RAPL file not found. Perhaps you are using a platform that does not support RAPL (e.g. Windows)")
            self.successful = False
        except PermissionError:
            logging.debug("PermissionError occured while reading RAPL file. Fix with \"sudo chmod -R a+r /sys/class/powercap/intel-rapl\"")
            self.successful = False

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            if self.successful:
                self.meter.begin()
                time.sleep(self.interval)
                self.meter.end()
                power = sum(map(lambda x: x / self.meter.result.duration, self.meter.result.pkg))
                self.measurements.append(power)
                self.timestamps.append(datetime.now())

    def after(self):
        if self.successful:
            self.data = {
                'timestamps' : self.timestamps,
                'measurements' : self.measurements,
                'interval' : self.interval
            }

    def log(self, benchmark):
        if self.successful:
            benchmark.log(self.description, self.measure_type, self.serialize(), unit='Watt')


class LatencyMetric(Metric):
    """The metric object to measure latency of the pipeline function

    To pass the number of data points processed, use method `track()`

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    """
    priority = 0
    measure_type = 'latency'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = self.num_entries / (after_time - self.before_time)

    def track(self, num_entries):
        """Sets the number of data points used to calculate latency by this object

        Parameters
        ----------
        num_entries: int
            the number of data points on which latency calculations are based
        """
        self.num_entries = num_entries

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='Seconds/entry')


class ThroughputMetric(Metric):
    """The metric object to measure throughput of the pipeline function

    To pass the number of data points processed, use method `track()`

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    """
    priority = 0
    measure_type = 'throughput'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = (after_time - self.before_time) / self.num_entries

    def track(self, num_entries):
        """Sets the number of data points used to calculate throughput by this object

        Parameters
        ----------
        num_entries: int
            the number of data points on which throughput calculations are based
        """
        self.num_entries = num_entries

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='Entries/second')


class CPUMetric(Metric):
    """The metric object to measure CPU usage of the running Python instance in percent

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between CPU usage measurements
    """
    priority = 1
    measure_type = 'cpu'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        self.process = psutil.Process(os.getpid())
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            self.measurements.append(self.process.cpu_percent(interval=self.interval))

    def after(self):
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit="%")
