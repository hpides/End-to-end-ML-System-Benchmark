FROM python:3.9

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

RUN git clone https://github.com/hpides/End-to-end-ML-System-Benchmark.git
RUN pip install --upgrade -e End-to-end-ML-System-Benchmark/umlaut/
RUN pip install "numpy>=1.20"
RUN pip install "pandas==2.0.2"
RUN pip install 'pygments<3.0.0,>=2.13.0'

WORKDIR /app

ENTRYPOINT ["/bin/bash"]