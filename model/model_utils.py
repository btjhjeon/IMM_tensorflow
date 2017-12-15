import os
import time
import numpy as np

import tensorflow as tf

import model.linear_layer as layers

def CopyLayers(sess, Lsc, Ltr = None, trainable_info=None):
    if trainable_info == None:
        trainable_info = np.tile([False], len(Lsc))

    for l in range(len(Lsc)):
        if len(Ltr) <= l:
            Ltr.append(layers.RegLinear(Lsc[l], trainable=trainable_info[l]))
        sess.run(Ltr[l].W.assign(Lsc[l].W.eval(sess)))
        sess.run(Ltr[l].b.assign(Lsc[l].b.eval(sess)))

    return Ltr

def CopyLayerValues(sess, Lsc, Ltr = None):
    if Ltr == None: Ltr = []
    for l in range(len(Lsc)):
        if len(Ltr) < l + 1: Ltr.append({})
        Ltr[l]['W'] = Lsc[l].W.eval(sess)
        Ltr[l]['b'] = Lsc[l].b.eval(sess)
    return Ltr

def ZeroLayers(sess, L):
    for l in range(len(L)):
        shape = L[l].W.get_shape().as_list()
        sess.run(L[l].W.assign(np.zeros(shape)))
        sess.run(L[l].b.assign(np.zeros(shape[-1])))

def AddLayers(sess, L1, L2, Ltr):
    op = []
    for l in xrange(len(L1)):
        v = sess.run([L1[l].W, L2[l].W, L1[l].b, L2[l].b])
        op += [Ltr[l].W.assign(v[0]+v[1]), Ltr[l].b.assign(v[2]+v[3])]
    sess.run(op)

def AddMultiTaskLayers(sess, Ls, Ltr, Lw, noOfTask):
    if len(Lw) < 1 or len(Ls) < 1:
        return

    ops = []
    noOfLayer = len(Lw[0])

    for i in range(noOfLayer):
        val_W = 0
        val_b = 0
        for j in range(noOfTask):
            val_W += Ls[j][i]['W'] * Lw[j][i]['W']
            val_b += Ls[j][i]['b'] * Lw[j][i]['b']

        ops.append(Ltr[i].W.assign(val_W))
        ops.append(Ltr[i].b.assign(val_b))

    sess.run(ops)

def Lw_maker(sess, L, alpha_info, adhoc_mnist=False):
    Layers_out = [];
    op = [];
    for l in range(len(L)):
        shape = L[l].W.get_shape().as_list()
        rl = layers.RegLinear(L[l])
        op += [rl.W.assign(np.zeros(shape)+alpha_info[l])]
        op += [rl.b.assign(np.zeros([shape[1],])+alpha_info[l])]
        Layers_out.append(rl)
    if adhoc_mnist: adhoc_MNIST_(sess,Layers_out)
    sess.run(op)
    return Layers_out

def CalculateWeighingBase(sess, L1, L2, alpha):
    Lw = CopyLayers(sess, L1, [])
    op = []
    for l in range(len(L1)):
        (W1, W2, b1, b2) = sess.run([L1[l].W, L2[l].W, L1[l].b, L2[l].b])
        op += [Lw[l].W.assign( alpha*W2 / ((1-alpha)*W1 + alpha*W2)) ]
        op += [Lw[l].b.assign( alpha*b2 / ((1-alpha)*b1 + alpha*b2)) ]
    sess.run(op)
    return Lw

def UpdateMultiTaskLwWithAlphas(L, alpha_list, noOfTask):
    Lw = []

    for i in range(noOfTask):
        if len(Lw) < i + 1:
            Lw.append([])
        for l in range(len(L)):
            if len(Lw[i]) < l + 1:
                Lw[i].append({})
            Lw[i][l]['W'] = np.zeros(L[l]['W'].shape) + alpha_list[i]
            Lw[i][l]['b'] = np.zeros(L[l]['b'].shape) + alpha_list[i]
    return Lw

def UpdateMultiTaskWeightWithAlphas(Fs, alpha_list, noOfTask):
    """
    Calculate Weight of Layers from Fisher Matrix

    Args:
        Fs: Fs[idxOfTask][idxOfLayer]['W' and 'b'] -> numpy
    Returns:
        Lw: Lw[idxOfTask][idxOfLayer]['W' and 'b'] -> numpy
    """
    Lw = []

    total_W = 0
    total_b = 0
    noOfLayer = len(Fs[0])

    # total: denominator
    total_W = []
    total_b = []
    for i in range(noOfLayer):
        total_W.append(alpha_list[noOfTask -1] * Fs[noOfTask - 1][i]['W'])
        total_b.append(alpha_list[noOfTask -1] * Fs[noOfTask - 1][i]['b'])
        for j in range(noOfTask - 1):
            total_W[i] += alpha_list[j] * Fs[j][i]['W']
            total_b[i] += alpha_list[j] * Fs[j][i]['b']

    # calculating layer weight
    for i in range(noOfTask):
        if len(Lw) < i + 1:
            Lw.append([])
        for j in range(noOfLayer):
            if len(Lw[i]) < j + 1:
                Lw[i].append({})
            val_W = alpha_list[i] * Fs[i][j]['W'] / total_W[j]
            val_b = alpha_list[i] * Fs[i][j]['b'] / total_b[j]
            Lw[i][j]['W'] = val_W
            Lw[i][j]['b'] = val_b

    return Lw

def PrintLayers(sess, L):
    print("Welcome to my PrintLayers.")
    for l in range(len(L)):
        shape = np.asarray(L[l].W.get_shape().as_list())
        (W_, b_) = sess.run([L[l].W, L[l].b])
        print("Level", l+1, ":", np.r_[shape, W_[0,0:2],W_[-1,-2:],b_[0],b_[-1]])
    print("")

def PrintLayers2(L):
    print("Welcome to my PrintLayers.")
    for l in range(len(L)):
        shape = L[l]['W'].shape
        W_, b_ = [L[l]['W'], L[l]['b']]
        print("Level", l+1, ":", np.r_[shape, W_[0,0:2],W_[-1,-2:],b_[0],b_[-1]])
    print("")
