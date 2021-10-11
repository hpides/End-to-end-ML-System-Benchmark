# End-to-end-ML-System-Benchmark
A modular suite for benchmarking all stages of Machine Learning pipelines. To find bottlenecks in such pipelines and compare different ML tools, this framework can calculate and visualize several metrics in the data preparation, model training, model validation and inference stages.


## Current metrics include:
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

## Current pipeline sources:
* Stock Market Prediction (https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f)
* MNIST Digit Recognition (https://github.com/rahulagg999/MNIST-Digit-Recognizer/blob/master/MNIST.ipynb)

## Documentation
* https://hpides.github.io/End-to-end-ML-System-Benchmark/

## How to use each decorator:
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
Needs a dict returned with an entry having the key "num_entries" containing the total amount of entries to be handled.
