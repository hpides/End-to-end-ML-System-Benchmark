FROM daphneeu/daphne-dev:latest_X86-64_CUDA

RUN apt-get update && apt-get install -y python3-pip

WORKDIR /app

RUN git clone https://github.com/hpides/End-to-end-ML-System-Benchmark.git
RUN pip install --upgrade -e End-to-end-ML-System-Benchmark/umlaut/
RUN pip install "numpy>=1.20"
RUN pip install "pandas==2.0.2"
RUN pip install 'pygments<3.0.0,>=2.13.0'

WORKDIR /app

# FOR THE KAI PIPELINE
RUN git clone -b dnn-ops https://github.com/corepointer/daphne.git
WORKDIR /app/daphne
RUN bash build.sh --clean --yes
RUN bash build.sh --cuda --installPrefix /usr/local --no-deps
RUN alias d='/app/daphne/bin/daphne'
WORKDIR /app/

ENTRYPOINT ["/bin/bash"]