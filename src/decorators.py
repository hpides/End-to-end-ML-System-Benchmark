import time
from config import parser as config


def measure_time(func):
    def inner(*args, **kwargs):
        before = time.perf_counter()
        result = func(*args, **kwargs)
        after = time.perf_counter()
        time_taken = after - before
        with open(config['filepaths']['out_file'], 'a+') as out_file:
            out_file.write(f"Time taken: {time_taken} \n")

        return result
    return inner
