- Download TP1.zip from https://daphne-eu.know-center.at/index.php/f/266532

- Unzip it to this source directory

- Create a virtual environment with 'python3 -m venv .venv'

- Activate the virtual environment with 'source .venv/bin/activate'

- Install pytorch depending on your system (e.g. https://pytorch.org/get-started/locally/)

- Install the rest of the dependencies with 'pip install scikit-learn tqdm pandas torchmetrics'

- Update line 92 with number of cpu cores on your system

- Optional: If you want to work with a shorter workload, update train2-control_stride.py by the following:

    * Decrease epoch count on line 95 e.g. to 1.

    * Uncomment lines 115, 116, 117 to skip additonal fold iterations

- Run 'python train2-control_stride.py'

- NB: This minimal python script setup was tested with Python version 3.10.6 and RHEL 7-
