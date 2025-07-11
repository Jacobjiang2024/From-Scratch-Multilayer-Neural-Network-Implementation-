import numpy as np
from Optimizer import Optimizer
from Lossfunction import *
'''
  Provide a common function

  classify: Convert the probability matrix output by softmax into predicted class labels.
  
  accuracy: Calculate the accuracy of the predicted results and the true labels.
  
  to_one_hot:Convert integer labels to one-hot encoding for softmax and cross entropy loss calculations.
'''

# ============Utility Functions ===============

def classify(probabilities):
    # The output probability matrix is converted into predicted class labels.
    return np.argmax(probabilities, axis=1)

def accuracy(predicted, target):
    # Calculate the accuracy of the predicted results and the true labels.
    return np.mean(predicted == target)

def to_one_hot(y, num_classes):
    # Create an all-zero matrix
    one_hot = np.zeros((len(y), num_classes))
    # It sets the class position corresponding to each sample in y to 1
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def clone_optimizer(opt):
    """Create a fresh copy of the optimizer with the same hyperparameters, but no internal state."""
    return Optimizer(
        mode=opt.mode,
        learning_rate=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
        beta1=opt.beta1,
        beta2=opt.beta2,
        epsilon=opt.epsilon
    )

# create network shape
def build_shape(input_dim, hidden_layers):
    return [input_dim] + hidden_layers + [10]

"""
    Calculating Micro F1 Scores for a Multi-Classification Task
    Parameters:
        y_true: numpy array, true labels (integers)
        y_pred: numpy array, predicted labels (integer)
        num_classes: total number of categories (e.g. 10 is 10)
    Returns:
        micro_f1: float
"""
# calculate Micro F1
def micro_f1_score(y_pred,y_true, num_classes):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Initialise total TP, FP, FN
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro precision = total_tp / (total_tp + total_fp)
    # Micro recall    = total_tp / (total_tp + total_fn)
    # Micro F1
    micro_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
    return micro_f1

# Recording result
def extract_metrics_from_history(name, history):
    val_accs = history['val_acc']
    best_epoch = int(np.argmax(val_accs)) + 1

    final_val_acc = val_accs[-1]
    final_train_acc = history['acc'][-1]
    final_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_f1 = history['val_f1'][-1]
    final_train_f1 = history['f1'][-1]

    metrics = {
        'Best Epoch': best_epoch,
        'Final Val Acc': round(final_val_acc, 4),
        'Train Acc': round(final_train_acc, 4),
        'Val Loss': round(final_val_loss, 4),
        'Train Loss': round(final_loss, 4),
        'Val F1': round(final_val_f1, 4),
        'Train F1': round(final_train_f1, 4),
        'Avg Training Time': round(np.mean(history['epoch_time']), 4)
    }

    return name, metrics

# calculate f1, loss and accuracy.
def calculate_result(pre, target):
    result = classify(pre[-1])
    f1 = micro_f1_score(result, target, num_classes=10)
    loss = cross_entropy_loss(pre[-1], target)
    acc = accuracy(result, target)  # calculate the accuracy
    return f1, acc, loss