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
import json
from torch import cuda
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage

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
    def __init__(self, metrics, benchmark, name=None):
        self.metrics = sorted(metrics)
        self.benchmark = benchmark
        self.method_name = name

    def __call__(self, func):
        def inner(*args, **kwargs):
            if self.method_name is None:
                self.method_name = func.__name__
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
            metric.log(self.benchmark, self.method_name)


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
        return self.data
        #return pickle.dumps(self.data)

    def log(self, benchmark, decorated_method_name):
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

    def log(self, benchmark, decorated_method_name):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='sec', method_name=decorated_method_name)

class GPUTimeMetric(Metric):
    """The metric object to measure the time taken for GPU execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    """
    priority = 0
    measure_type = 'gputime'
    needs_threading = False

    def before(self):
        self.start_event = cuda.Event(enable_timing=True)
        self.end_event = cuda.Event(enable_timing=True)
        self.start_event.record()

    def after(self):
        self.end_event.record()
        cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)  # Time in milliseconds
        self.data = elapsed_time / 1000.0  # Convert to seconds

    def log(self, benchmark, decorated_method_name):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='sec', method_name=decorated_method_name)

class GPUMemoryMetric(Metric):
    """The metric object to measure GPU memory used in the execution

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between memory measurements
    """
    priority = 3
    measure_type = 'gpumemory'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            self.measurements.append(nvmlDeviceGetMemoryInfo(self.handle).used / (2 ** 20))
            time.sleep(self.interval)

    def after(self):
        nvmlShutdown()
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark, decorated_method_name):
        json_data = json.dumps(self.serialize(), indent=4, default=str)
        benchmark.log(self.description, self.measure_type, json_data, unit="MiB", method_name=decorated_method_name)

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
            try:
                self.timestamps.append(datetime.now())
                self.measurements.append(self.process.memory_info().rss / (2 ** 20))
                time.sleep(self.interval)
            except psutil.NoSuchProcess:
                finish_event.set()
                break

    def after(self):
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark, decorated_method_name):
        json_data = json.dumps(self.serialize(), indent=4, default=str)
        benchmark.log(self.description, self.measure_type, json_data, unit="MiB", method_name=decorated_method_name)
        
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

    def log(self, benchmark, decorated_method_name):
        if self.successful:
            json_data = json.dumps(self.serialize(), indent=4, default=str)
            benchmark.log(self.description, self.measure_type, json_data, unit='µJ', method_name=decorated_method_name)

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

    def log(self, benchmark, decorated_method_name):
        if self.successful:
            json_data = json.dumps(self.serialize(), indent=4, default=str)
            benchmark.log(self.description, self.measure_type, json_data, unit='Watt', method_name=decorated_method_name)

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
        #self.data = str([(after_time - self.before_time) / self.num_entries, self.num_entries])
        self.data = (after_time - self.before_time) / self.num_entries


    def track(self, num_entries):
        """Sets the number of data points used to calculate latency by this object

        Parameters
        ----------
        num_entries: int
            the number of data points on which latency calculations are based
        """
        self.num_entries = num_entries

    def log(self, benchmark, decorated_method_name):
        benchmark.log(str(self.description + "\n (#Entries = " + str(self.num_entries) + ")"), self.measure_type, self.serialize(), unit='Seconds/entry', method_name=decorated_method_name)

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
        self.data = self.num_entries / (after_time - self.before_time)

    def track(self, num_entries):
        """Sets the number of data points used to calculate throughput by this object

        Parameters
        ----------
        num_entries: int
            the number of data points on which throughput calculations are based
        """
        self.num_entries = num_entries

    def log(self, benchmark, decorated_method_name):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='Entries/second', method_name=decorated_method_name)

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
            try:
                self.measurements.append(self.process.cpu_percent(interval=self.interval))
                self.timestamps.append(datetime.now())
            except psutil.NoSuchProcess:
                finish_event.set()
                break

    def after(self):
        self.data = {
            'timestamps': self.timestamps[2:],
            'measurements': self.measurements[2:]
        }

    def log(self, benchmark, decorated_method_name):
        json_data = json.dumps(self.serialize(), indent=4, default=str)
        benchmark.log(self.description, self.measure_type, json_data, unit="%", method_name=decorated_method_name)

class GPUMetric(Metric):
    """The metric object to measure GPU usage of the running instance in percent

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between GPU usage measurements
    """
    priority = 1
    measure_type = 'gpu'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)  # You may want to parameterize the GPU index
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            utilization = nvmlDeviceGetUtilizationRates(self.handle)
            self.measurements.append(utilization.gpu)
            finish_event.wait(self.interval)  # This is better for clean exit

    def after(self):
        nvmlShutdown()
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark, decorated_method_name):
        json_data = json.dumps(self.serialize(), indent=4, default=str)
        benchmark.log(self.description, self.measure_type, json_data, unit="%", method_name=decorated_method_name)

class GPUPowerMetric(Metric):
    """The metric object to measure GPU power usage of the running instance in watts

    Parameters
    ----------
    description: str
        The description of this metric and function which is added to the database
    interval: int, default=1
        The number of seconds between GPU power measurements
    """
    priority = 1
    measure_type = 'gpupower'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval

    def before(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)  # You may want to parameterize the GPU index
        self.timestamps = []
        self.measurements = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.timestamps.append(datetime.now())
            power_usage = nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert from milliwatts to watts
            self.measurements.append(power_usage)
            finish_event.wait(self.interval)  # This is better for clean exit

    def after(self):
        nvmlShutdown()
        self.data = {
            'timestamps': self.timestamps,
            'measurements': self.measurements
        }

    def log(self, benchmark, decorated_method_name):
        json_data = json.dumps(self.serialize(), indent=4, default=str)
        benchmark.log(self.description, self.measure_type, json_data, unit="W", method_name=decorated_method_name)

