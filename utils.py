import numpy as np

def compute_metrics(eval_dataset):
    predictions, labels = eval_dataset
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(preictions=predictions, references=labels)

