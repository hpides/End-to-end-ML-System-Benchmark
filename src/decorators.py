import time


class MeasureTime(object):
    def __init__(self, output_file):
        self.output_file = output_file

    def __call__(self, func):
        def inner(*args, **kwargs):
            before = time.perf_counter()
            result = func(*args, **kwargs)
            after = time.perf_counter()
            time_taken = after - before
            with open(self.output_file, 'a+') as out_file:
                out_file.write(f"Time taken: {time_taken} \n")
            return result
        return inner