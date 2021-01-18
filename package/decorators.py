import time
import resource
from concurrent.futures import ThreadPoolExecutor

class Measure(object):
    name = None

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def __call__(self, func):
        self.func_name = func.__name__


class MeasureTime(Measure):
    measurement_type = "Time"

    def __call__(self, func):
        super().__call__(func)

        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            self.benchmark.log(func.__name__, self.measurement_type, time_taken)
            return result

        return inner


class MeasureMemorySamples(Measure):
    measurement_type = "Memory"

    def __init__(self, benchmark, interval):
        super().__init__(benchmark)
        self.interval = interval
        self.keep_measuring = True

    def __call__(self, func):
        super().__call__(func)
        self.func_name = func.__name__

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
            measurement_value = f"{round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)} MB"
            self.benchmark.log(self.func_name, self.measurement_type, measurement_value)
            time.sleep(self.interval)


class MeasureConfusion(Measure):
    measurement_type = "Confusion"

    ## Erwartet ein Dict mit den Metriken als return type

    def __call__(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)

            if type(result) is not dict:
                raise TypeError(f"Illegal argument type. Expected dict. Got {type(result)}")

            if "TP" in result and "TN" in result and "FP" in result and "FN" in result and "0_class" not in result and "1_class" not in result:
                result["0_class"] = result["TN"] + result["FP"]
                result["1_class"] = result["TP"] + result["FN"]

            if "TP" in result and ("FN" in result or "1_class" in result) and "recall" not in result:
                if "FN" in result:
                    result["recall"] = result["TP"] / (result["TP"] + result["FN"])
                else:
                    result["recall"] = result["TP"] / result["1_class"]

            if "TP" in result and "FP" in result and "precision" not in result:
                result["precision"] = result["TP"] / (result["TP"] + result["FP"])

            if "1_class" in result and "0_class" in result and "recall" in result and "precision" in result:
                if "TP" not in result:
                    result["TP"] = result["recall"] * result["1_class"]
                if "FN" not in result:
                    result["FN"] = result["1_class"] - result["TP"]
                if "FP" not in result:
                    result["FP"] = (result["precision"] / result["TP"]) ** -1 - result["TP"]
                if "TN" not in result:
                    result["TN"] = result["0_class"] - result["FP"]

            if "TP" in result and "TN" in result and "0_class" in result and "1_class" in result and "accuracy" not in result:
                result["accuracy"] = (result["TP"] + result["TN"]) / (result["0_class"] + result["1_class"])

            if "TP" in result and "FP" in result and "FN" in result and "f1_score" not in result:
                result["f1_score"] = (result["TP"] * 2) / (result["TP"] * 2 + result["FP"] + result["FN"])

            self.benchmark.log(func.__name__, self.measurement_type, str(result))
            return result

        return inner


class MeasureLatency(Measure):
    measurement_type = "Latency"

    def __init__(self, benchmark, number_of_entries):
        super().__init__(benchmark)
        self.number_of_entries = number_of_entries

    def __call__(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            before = time.perf_counter()
            after = time.perf_counter()
            time_taken = after - before
            # number_of_entries = 40000000
            latency = time_taken / self.number_of_entries
            self.benchmark.log(func.__name__, self.measurement_type, latency)
            return result

        return inner


class MeasureThroughput(Measure):
    measurement_type = "Throughput"

    def __init__(self, benchmark, number_of_entries):
        super().__init__(benchmark)
        self.number_of_entries = number_of_entries

    def __call__(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            before = time.perf_counter()
            after = time.perf_counter()
            time_taken = after - before
            # number_of_entries = 40000000
            throughput = self.number_of_entries / time_taken
            self.benchmark.log(func.__name__, self.measurement_type, throughput)
            return result

        return inner
