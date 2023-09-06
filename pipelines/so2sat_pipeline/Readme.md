# Steps to rerun the pipeline

## Download datasets
- the datasets are available under
  - https://dataserv.ub.tum.de/index.php/s/m1483140
- you can download them using wget
  - e.g.: `wget https://dataserv.ub.tum.de/s/m1483140/download?path=%2F&files=validation.h5 --no-check-certificate`
  - after download rename using `mv` cmd to have names like `validation.h5`

## create smaller datasets
- to debug and also showcase umlaut on smaller datasets us the script [generate_subset_data.py](generate_subset_data.py)
- for now, we use three datasets per split
  - full dataset
  - 320 samples
  - 32 samples

## Requirements
- you can find a requirements that are compatible with UMLAUT and the pipeline under
  - [requirements-updated.txt](requirements-updated.txt)

- we uploaded a snapshot of our dev container to persist the environment
  - [link to container](https://hub.docker.com/repository/docker/slin96/umlaut-so2sat/tags?page=1&ordering=last_updated)  
  - **activate the env by**: `source /home/daphneusr/.virtualenvs/End-to-end-ML-System-Benchmark/bin/activate`