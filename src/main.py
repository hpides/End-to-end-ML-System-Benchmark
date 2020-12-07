from decorators import MeasureTime
from config import parser as config


@MeasureTime(config['filepaths']['out_file'])
def print_this(string, times):
    for i in range(times):
        print(string)


def main():
    print_this("hello", 5)


if __name__ == "__main__":
    main()
