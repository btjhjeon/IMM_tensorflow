import os
import time
import numpy as np


def SetDefaultAsNatural(FLAGS):
    if hasattr(FLAGS, 'epoch') and FLAGS.epoch < 0:
        FLAGS.epoch = 60

    if hasattr(FLAGS, 'learning_rate') and FLAGS.learning_rate < 0:
        FLAGS.learning_rate = 0.1

    if hasattr(FLAGS, 'regularizer') and FLAGS.regularizer < 0:
        FLAGS.regularizer = 1e-4

    if hasattr(FLAGS, 'alpha') and FLAGS.alpha < 0:
        FLAGS.alpha = 1.0 / 3

def PrintResults(alpha, results):
    """
    print accuracy results.

    Args:
        results: list of accuracy results.
            the half size of list is for training accuracy
            and the other is for test accuracy.
    """
    result_text = "%.2f" % alpha
    for i in range(int(len(results)/2)):
        result_text += ", train-idx%d: %.4f" % (i+1, results[i])
    for i in range(int(len(results)/2)):
        result_text += ", test-idx%d: %.4f" % (i+1, results[i + int(len(results)/2)])
    print(result_text)
