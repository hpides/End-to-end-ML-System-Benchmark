# UMLAUT (Universal Machine Learning Analysis UTility)

A modular suite for benchmarking all stages of Machine Learning pipelines. To find bottlenecks in such pipelines and compare different ML tools, this framework can calculate and visualize several metrics in the data preparation, model training, model validation and inference stages.

## Installation & Setup

Clone the current repository with the following command:

```
git clone git@github.com:hpides/End-to-end-ML-System-Benchmark.git
```
To use the package in a Python project, include it in the *requirements.txt* file.  
That can be done with a path reference to the local repository. Include the following line in your requirements file, with your local path.

```
-e <PATH_TO_REPOSITORY>/umlaut/
```

Or, through **pip**:

```
pip install -e <PATH_TO_REPOSITORY>/umlaut/
```



## System Integration

Upon installation, UMLAUT can be imported in any Python pipeline. 
To import UMLAUT, use the following *import* statement in the Python script.

``` 
import umlaut 
```

To intialize a benchmark, initialize an instance of the *Benchmark* class. It requires two string parameters, *db_file*, and *description (optional)*.

``` 
from umlaut import Benchmark

benchmark = Benchmark(db_file = 'hello_world.db', description = 'Measurements for benchmarking hello_world.py pipeline.')
```

## Comand Line Interface

Measurements are accessed through UMLAUT's CLI tool. It can be invoked from a *bash* terminal with the following command.

```
umlaut-cli <db_file>
```

To read through the measurements from the *hello_world.db* database, we insert the *db_file* name in the command.

```
umlaut-cli hello_world.db
```

For detailed descriptions of all avaiable arguments and flags, call the *help* command for *umlaut-cli*.

```
umlaut-cli --help
```

## Metrics

UMLAUT collects measurements of the following metrics:  

* Time spent
* Memory usage
* Loss (single run and multiple runs)
* Influence of batch size and #epochs
* Influence of learning rate
* Time to Accuracy (single run and multiple runs)
* Power usage
* Multiclass Confusion Matrix
* Standard metrics as accuracy, F1, TP/TN etc.
* Latency
* Throughput

## Visualization

Through the CLI tool, the measurements for each of the metrics can be visualized. For each pipeline, users can generate plots for one or more metrics.   
Measurements for the same metric for multiple pipelines can be shown on a single plot. Examples of using the CLI toolkit for visualization are shown below. 

![Selecting single pipeline to visualize](plots/umlaut_cli_single_pipeline_1.png)  

![Selecting measurements for a single pipeline](plots/umlaut_cli_single_pipeline_2.png)

![Results for CPU Usage [single pipeline]](plots/cpu_single_pipeline.png)

![Results for Memory Usage [single pipeline]](plots/memory_single_pipeline.png)    



![Selecting multiple pipelines to visualize](plots/umlaut_cli_3pipelines_1.png)  

![Selecting measurements for multiple pipelines](plots/umlaut_cli_3pipelines_2.png)

![Results for CPU Usage [multiple pipelines]](plots/cpu_3pipelines.png)

![Results for Power Usage [multiple pipelines]](plots/power_3pipeline_runs.png)

## Example Pipelines
In the *pipelines* folder, there are several examples of the following pipelines where UMLAUT is integrated. 

* So2Sat Earth Observation (https://scihub.copernicus.eu/)
* Backblaze Hard Drive Anomaly Prediction (https://www.backblaze.com/b2/hard-drive-test-data.html)
* Stock Market Prediction (https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f)
* MNIST Digit Recognition (https://github.com/rahulagg999/MNIST-Digit-Recognizer/blob/master/MNIST.ipynb)

## Documentation
* https://hpides.github.io/End-to-end-ML-System-Benchmark/

<!-- ## How to use each decorator:
### Following Decorators only need to be applied to a method:
* Time
* Memory
* Energy

### TimeToAccuracyMult
Needs a dict returned with an entry having the key "accuracy" containing the accuracy value (#epochs currently hardcoded, needs to be editable by user). Use this when the user is unable to keep track of the accuracy in each epoch. Reruns the method multiple times. Slower than alternative below.

### TimeToAccuracy
Needs a dict returned with an entry having the key "accuracy" containing a list of accuracies for each epoch. Use this when the user can track the accuracy after each epoch. Runs the method once. Faster than alternative above.

### BatchSizeInfluence
Needs a dict returned with an entry having the key "loss" containing a list of losses for each epoch. Needs the decorated method to have an optional "batch_size" parameter (#batches currently hardcoded, needs to be editable by user).

### BatchAndEpochInfluenceMult
Needs a dict returned with an entry having the key "loss" containing the loss value. Needs the decorated method to have an optional "batch_size" and "epochs" parameter (#batches and #epochs currently hardcoded, needs to be editable by user). Reruns multiple times (see TTAMult).


### BatchAndEpochInfluence
Needs a dict returned with an entry having the key "loss" containing a list of losses for each epoch. Needs the decorated method to have an optional "batch_size" and "epochs" parameter (#batches and #epochs currently hardcoded, needs to be editable by user). Runs once (see TTA).

### LossMult
Needs a dict returned with an entry having the key "loss" containing the loss value. Reruns multiple times (see TTAMult).

### Loss
Needs a dict returned with an entry having the key "loss" containing a list of losses for each epoch. Runs once (see TTA).

### LearningRate
Needs a dict returned with an entry having the key "loss" containing a list of losses for each epoch.

### Confusion
Needs a dict returned with entries having the key named after basic metrics containing the corresponding values (0_class, 1_class, TN, TF, FN, FP, recall, precision, accuracy, f1_score).

### MulticlassConfusion
Needs a dict returned with an entry having the key "confusion_matrix" containing a list of the confusion matrix entries and "classes" containing a list of the classes in said confusion matrix.

### MulticlassConfusionTF
Same as above, but for tensorflow. User can pass on the confusion matrix without unpacking the values out of the tensors (done inside the decorator).

### Latency
Needs a dict returned with an entry having the key "num_entries" containing the total amount of entries to be handled.

### Throughput
Needs a dict returned with an entry having the key "num_entries" containing the total amount of entries to be handled. -->
