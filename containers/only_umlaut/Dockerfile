FROM python:3.9

RUN git clone --single-branch --branch=experimental https://github.com/hpides/End-to-end-ML-System-Benchmark.git
RUN pip install -e End-to-end-ML-System-Benchmark/umlaut/
WORKDIR End-to-end-ML-System-Benchmark
COPY run_script.py .
ENTRYPOINT ["python", "-u", "run_script.py"]