import time
import resource
from concurrent.futures import ThreadPoolExecutor
import threading
import json

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

    def __init__(self, benchmark, description, interval=0.1):
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

            self.benchmark.log(self.description, self.measurement_type, str(result))
            return result

        return inner


class MeasureMulticlassConfusion(Measure):
    measurement_type = "Multiclass Confusion Matrix"

    def __call__(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            for i in range(len(result["confusion matrix"])):
                self.benchmark.log(self.description, "Multiclass Confusion Matrix Class", result["classes"][i])
                for j in range((len(result["confusion matrix"]))):
                    self.benchmark.log(self.description, self.measurement_type, str(result["confusion matrix"][i][j]))

            result["confusion matrix"] = str(result["confusion matrix"])
            result["classes"] = str(result["classes"])
            return result

        return inner


class MeasureLatency(Measure):
    measurement_type = "Latency"

    def __call__(self, func):
        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            latency = result['num_entries'] / time_taken
            self.benchmark.log(self.description, self.measurement_type, latency, "entries per second")
            return result

        return inner


class MeasureThroughput(Measure):
    measurement_type = "Throughput"

    def __call__(self, func):
        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            latency = time_taken / result['num_entries']
            self.benchmark.log(self.description, self.measurement_type, latency, "seconds per entry")
            return result

        return inner
