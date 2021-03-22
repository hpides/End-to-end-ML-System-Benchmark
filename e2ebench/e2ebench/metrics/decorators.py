
class Measure(object):
    name = None

    def __init__(self, benchmark, description):
        self.benchmark = benchmark
        self.description = description
# multiple run version. Should only be used if version below doesn't work with the chosen training algorithm
# (i.e. cant return accuracy for each epoch)
class MeasureTimeToAccuracyMult(Measure):
    measurement_type = "TTA"

    def __call__(self, func):
        def inner(*args, **kwargs):
            finalResult = None
            for i in range(1,11):                                       # no. of epochs should be choosable by the user
                result = func(*args, epochs=i, **kwargs)
                accuracy = result["accuracy"]
                self.benchmark.log(self.description, self.measurement_type, accuracy)
                finalResult = result
            return finalResult
        return inner


# single run version. more efficient. needs array of accuracies for each epoch)
class MeasureTimeToAccuracy(Measure):
    measurement_type = "TTA"

    def __call__(self, func):
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            accuracy = result["accuracy"]
            for i in range(len(accuracy)):
                self.benchmark.log(self.description, self.measurement_type, accuracy[i])
            return result
        return inner


# class MeasureMetricsPerEpoch(Measure):
#    measurement_type = "Metrics per Epoch"
#
#    def __call__(self, func):
#        def inner(*args, **kwargs):
#            result = func(*args, **kwargs)
#            metrics = result["metrics"]
#            for metric in metrics:
#                for value in metrics[metric]:
#                    self.benchmark.log(self.description, self.measurement_type + metric, value)
#            return result
#        return inner


class MeasureBatchSizeInfluence(Measure):
    measurement_type = "Batch"

    def __call__(self, func):
        def inner(*args, **kwargs):
            finalResult = None
            for i in range(1, 11):                                       # batch range should be choosable by the user
                result = func(*args, batch_size=2**i, **kwargs)
                loss = result["loss"][-1]
                self.benchmark.log(self.description, self.measurement_type, loss)
                finalResult = result
            return finalResult
        return inner


class MeasureBatchAndEpochInfluenceMult(Measure):
    measurement_type = "Batch and Epoch"

    def __call__(self, func):
        def inner(*args, **kwargs):
            finalResult = None
            for i in range(1, 11):                                       # batch range should be choosable by the user
                for j in range(1, 11):                                   # no. of epochs should be choosable by the user
                    result = func(*args, epochs=i, batch_size=2 ** j, **kwargs)
                    loss = result["loss"][-1]
                    self.benchmark.log(self.description, self.measurement_type, loss)
                    finalResult = result
            return finalResult
        return inner


class MeasureBatchAndEpochInfluence(Measure):
    measurement_type = "Batch and Epoch"

    def __call__(self, func):
        def inner(*args, **kwargs):
            finalResult = None
            for j in range(1, 11):                                   # batch range should be choosable by the user
                result = func(*args, epochs=10, batch_size=2 ** j, **kwargs)    # no. of epochs should be choosable by the user
                loss = result["loss"]
                for i in range(len(loss)):
                    self.benchmark.log(self.description, self.measurement_type, loss[i])
                finalResult = result
            return finalResult
        return inner



class MeasureLearningRate(Measure):
    measurement_type = "Learning Rate"

    def __call__(self, func):
        def inner(*args, **kwargs):
            for i in range(0, 45):
                result = func(*args, lr=(10**floor(i/9)*((i % 9)+1)*0.00001), **kwargs)
                loss = result["loss"][-1]
                self.benchmark.log(self.description, self.measurement_type, loss)
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
