import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_dataset):
    predictions, labels = eval_dataset
    predictions = np.argmax(predictions, axis=1)    
    metric.add_batch(predictions=predictions, references=labels)
    return metric.compute()

