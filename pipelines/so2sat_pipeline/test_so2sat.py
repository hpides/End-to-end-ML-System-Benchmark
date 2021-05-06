import h5py
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
import e2ebench
from benchmarking import bm
import e2ebench as eb


@eb.BenchmarkSupervisor([eb.MemoryMetric('test memory'), eb.TimeMetric('test time')], bm)
def test(model):
    cmt = eb.ConfusionMatrixTracker(bm)

    # n = 32768  # 2**15
    n = 1024
    img_width, img_height, img_num_channels = 32, 32, 8

    f = h5py.File('data/testing.h5', 'r')
    input_test = f['sen1'][0:n]
    label_test = f['label'][0:n]
    f.close()

    input_test = input_test.reshape((len(input_test), img_width, img_height, img_num_channels))

    # Generate generalization metrics
    score = model.evaluate(input_test, label_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    # Generate confusion matrix
    pred_test = model.predict_classes(input_test)
    label_test = np.argmax(label_test, axis=1)

    con_mat = confusion_matrix(label_test, pred_test)

    print("Confusion Matrix: \n")
    print(con_mat)

    classes = ["compact high-rise", "compact mid-rise", "compact low-rise",
               "open high-rise", "open mid-rise", "open low-rise",
               "lightweight low-rise", "large low-rise", "sparsely built",
               "heavy industry", "dense trees", "scattered tree",
               "brush, scrub", "low plants", "bare rock or paved",
               "bare soil or sand", "water"]

    cmt.track(con_mat, classes, "confusion matrix")

    return {"confusion matrix": con_mat, "classes": classes}
