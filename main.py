import time
import argparse
import numpy as np
import tensorflow as tf

import preprocess.mnist as preprocess
import utils
from model import model_utils
from model import imm


print("==> parsing input arguments")
flags = tf.app.flags

## Data input settings
flags.DEFINE_boolean("mean_imm", True, "include Mean-IMM")
flags.DEFINE_boolean("mode_imm", True, "include Mode-IMM")

## Model Hyperparameter 
flags.DEFINE_float("alpha", -1, "alpha(K) of Mean & Mode IMM (cf. equation (3)~(8) in the article)")

## Training Hyperparameter
flags.DEFINE_float("epoch", -1, "the number of training epoch")
flags.DEFINE_string("optimizer", 'SGD', "the method name of optimization. (SGD|Adam|Momentum)")
flags.DEFINE_float("learning_rate", -1, "learning rate of optimizer")
flags.DEFINE_integer("batch_size", 50, "mini batch size")

FLAGS = flags.FLAGS
utils.SetDefaultAsNatural(FLAGS)


mean_imm = FLAGS.mean_imm
mode_imm = FLAGS.mode_imm
alpha = FLAGS.alpha
optimizer = FLAGS.optimizer
learning_rate = FLAGS.learning_rate
epoch = FLAGS.epoch
batch_size = FLAGS.batch_size

no_of_task = 3
no_of_node = [784,800,800,10]
keep_prob_info = [0.8, 0.5, 0.5]


# data preprocessing
x, y, x_, y_, xyc_info = preprocess.XycPackage()

start = time.time()

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    mlp = imm.TransferNN(no_of_node, (optimizer, learning_rate), keep_prob_info=keep_prob_info)

    sess.run(tf.global_variables_initializer())

    L_copy = []
    FM = []
    for i in range(no_of_task):
        print("")
        print("================= Train task #%d (%s) ================" % (i+1, optimizer))

        mlp.Train(sess, x[i], y[i], x_[i], y_[i], epoch, mb=batch_size)
        mlp.Test(sess, [[x[i],y[i]," train"], [x_[i],y_[i]," test"]])

        if mean_imm or mode_imm:
            L_copy.append(model_utils.CopyLayerValues(sess, mlp.Layers))
        if mode_imm:
            FM.append(mlp.CalculateFisherMatrix(sess, x[i], y[i]))

    mlp.TestAllTasks(sess, x_, y_)


    alpha_list = [(1-alpha)/(no_of_task-1)] * (no_of_task-1)
    alpha_list.append(alpha)
    ######################### Mean-IMM ##########################
    if mean_imm:
        print("")
        print("Main experiment on %s + Mean-IMM, shuffled MNIST" % optimizer)
        print("============== Train task #%d (Mean-IMM) ==============" % no_of_task)

        LW = model_utils.UpdateMultiTaskLwWithAlphas(L_copy[0], alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
        ret = mlp.TestTasks(sess, x, y, x_, y_, debug = False)
        utils.PrintResults(alpha, ret)

        mlp.TestAllTasks(sess, x_, y_)

    ######################### Mode-IMM ##########################
    if mode_imm:
        print("")
        print("Main experiment on %s + Mode-IMM, shuffled MNIST" % optimizer)
        print("============== Train task #%d (Mode-IMM) ==============" % no_of_task)

        LW = model_utils.UpdateMultiTaskWeightWithAlphas(FM, alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
        ret = mlp.TestTasks(sess, x, y, x_, y_, debug = False)
        utils.PrintResults(alpha, ret)

        mlp.TestAllTasks(sess, x_, y_)

    print("")
    print("Time: %.4f s" % (time.time()-start))
