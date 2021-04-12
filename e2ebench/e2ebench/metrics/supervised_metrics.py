from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import pickle
import psutil
import threading
import time
import pyRAPL
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


class TimeVisualizer:
    def __init__(self, serialized_bytes):
        self.data = pickle.loads(serialized_bytes)

    def visualize(self, uuid, description):

        dic = {"uuid": [uuid]}
        dic[description] = self.data

        df = pd.DataFrame(
            dic,
            index=[uuid]
        )

        ax = df.plot.barh(stacked=False)
        plt.title("Time spent in phases")
        plt.xlabel("Time in seconds")

        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        plt.show()


class MemoryMetric(Metric):
    priority = 3
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


class MemoryVisualizer:
    def __init__(self, serialized_bytes):
        deserialized = pickle.loads(serialized_bytes)
        self.timestamps = deserialized["timestamps"]
        self.measurements = deserialized["measurements"]

    def visualize(self, uuid, description):

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        ax.plot(self.timestamps,
                self.measurements,
                label=("Run from " + uuid))

        ax.set_ylabel("MB used")
        ax.set_xlabel("Time in seconds")
        plt.legend(loc=2)
        plt.title("Memory usage")

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.show()


class EnergyMetric(Metric):
    priority = 1
    measure_type = 'energy'
    needs_threading = False

    def before(self):
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')
        self.meter.begin()

    def after(self):
        self.meter.end()
        self.data = sum(self.meter.result.pkg)

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='µJ')


class EnergyVisualizer:
    def __init__(self, serialized_bytes):
        self.data = pickle.loads(serialized_bytes)

    def visualize(self, uuid, description):

        dic = {"uuid": [uuid]}
        dic[description] = self.data

        df = pd.DataFrame(
            dic,
            index=[uuid]
        )

        ax = df.plot.barh(stacked=False)
        plt.title("Power used during run")
        plt.xlabel("µJ used")

        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        plt.show()


class PowerMetric(Metric):
    priority = 3
    measure_type = 'power'
    needs_threading = True

    def __init__(self, description, interval=1):
        super().__init__(description)
        self.interval = interval
        self.meter = None

    def before(self):
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement('bar')
        self.measurements = []
        self.timestamps = []

    def meanwhile(self, finish_event):
        while not finish_event.isSet():
            self.meter.begin()
            time.sleep(self.interval)
            self.meter.end()
            power = sum(map(lambda x: x / self.meter.result.duration, self.meter.result.pkg))
            self.measurements.append(power)
            self.timestamps.append(datetime.now())

    def after(self):
        self.data = {
            'timestamps' : self.timestamps,
            'measurements' : self.measurements
        }

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize())


class PowerVisualizer:
    def __init__(self, serialized_bytes):
        deserialized = pickle.loads(serialized_bytes)
        self.timestamps = deserialized["timestamps"]
        self.measurements = deserialized["measurements"]

    def visualize(self, uuid, description):

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        ax.plot(self.timestamps,
                self.measurements,
                label=("Run from " + uuid))

        ax.set_ylabel("Watt used")
        ax.set_xlabel("Time in seconds")
        plt.legend(loc=2)
        plt.title("Power consumption")

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.show()


class LatencyMetric(Metric):
    priority = 0
    measure_type = 'latency'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = self.num_entries / (after_time - self.before_time)

    def track(self, num_entries):
        self.num_entries = num_entries

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='entries/second')


class LatencyVisualizer:
    def __init__(self, serialized_bytes):
        self.data = pickle.loads(serialized_bytes)

    def visualize(self, uuid, description):

        dic = {"uuid": [uuid]}
        dic[description] = self.data

        df = pd.DataFrame(
            dic,
            index=[uuid]
        )

        ax = df.plot.barh(stacked=False)
        plt.title("Latency")
        plt.xlabel("Entries per second")

        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        plt.show()


class ThroughputMetric(Metric):
    priority = 0
    measure_type = 'throughput'
    needs_threading = False

    def before(self):
        self.before_time = time.perf_counter()

    def after(self):
        after_time = time.perf_counter()
        self.data = (after_time - self.before_time) / self.num_entries

    def track(self, num_entries):
        self.num_entries = num_entries

    def log(self, benchmark):
        benchmark.log(self.description, self.measure_type, self.serialize(), unit='seconds/entry')


class ThroughputVisualizer:
    def __init__(self, serialized_bytes):
        self.data = pickle.loads(serialized_bytes)

    def visualize(self, uuid, description):

        dic = {"uuid": [uuid]}
        dic[description] = self.data

        df = pd.DataFrame(
            dic,
            index=[uuid]
        )

        ax = df.plot.barh(stacked=False)
        plt.title("Throughput")
        plt.xlabel("Seconds per entry")

        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        plt.show()
