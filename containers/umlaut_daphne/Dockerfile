FROM daphneeu/daphne-dev:latest_X86-64_BASE 

WORKDIR /app

# DOWNLOAD THE BINARY
# RUN wget -qO- https://raw.githubusercontent.com/daphne-eu/daphne/main/KEYS.txt | gpg --import
# RUN wget https://github.com/daphne-eu/daphne/releases/download/0.2/daphne-X86-64-v0.2-bin.tgz.sha512sum
# RUN wget https://github.com/daphne-eu/daphne/releases/download/0.2/daphne-X86-64-v0.2-bin.tgz.asc
# RUN wget https://github.com/daphne-eu/daphne/releases/download/0.2/daphne-X86-64-v0.2-bin.tgz
# RUN sha512sum -c daphne-X86-64-v0.2-bin.tgz.sha512sum
# RUN gpg --verify daphne-X86-64-v0.2-bin.tgz.asc
# RUN tar xf daphne-X86-64-v0.2-bin.tgz
# # add lib directory to environment ENV LD_LIBRARY_PATH="/app/daphne-X86-64-v0.2-bin/lib:$LD_LIBRARY_PATH"

# ENV LD_LIBRARY_PATH="/app/daphne-X86-64-v0.2-bin/lib"

# WORKDIR /app/daphne-X86-64-v0.2-bin
# ENV DAPHNE_ROOT="/app/daphne-X86-64-v0.2-bin"
# ENV LD_LIBRARY_PATH="$DAPHNE_ROOT/lib:$DAPHNE_ROOT/thirdparty/installed/lib:$LD_LIBRARY_PATH"
# ENV PYTHONPATH="$PYTHONPATH:$DAPHNE_ROOT/src/"

RUN apt-get update && apt-get install -y python3-pip


WORKDIR /app

RUN git clone https://github.com/hpides/End-to-end-ML-System-Benchmark.git
RUN pip install --upgrade -e End-to-end-ML-System-Benchmark/umlaut/
RUN pip install "numpy>=1.20"
RUN pip install "pandas==2.0.2"
RUN pip install 'pygments<3.0.0,>=2.13.0'

WORKDIR /app

# FOR THE KAI PIPELINE
RUN git clone https://github.com/daphne-eu/daphne.git
WORKDIR /app/daphne
RUN bash build.sh --clean --yes
RUN bash build.sh --no-deps --installPrefix /usr/local

WORKDIR /app/

ENTRYPOINT ["/bin/bash"]