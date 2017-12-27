Overcoming Catastrophic Forgetting by Incremental Moment Matching (IMM)
===================================

IMM incrementally matches the moment of the posterior distribution of neural networks, which is trained for the first and the second task respectively. The experiment is only for the shuffled MNIST task.

## Reference

[1] [Sang-Woo Lee, Jin-Hwa Kim, Jaehyun Jun, Jung-Woo Ha and Byoung-Tak Zhang. Overcoming Catastrophic Forgetting by Incremental Moment Matching. In Advances in Neural Information Processing Systems, 2017.](https://arxiv.org/abs/1703.08475)

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

    @inproceedings{Lee2017,
    author = {Lee, Sang-Woo Lee and Kim, Jin-Hwa and Jun, Jaehyun and Ha, Jung-Woo and Zhang, Byoung-Tak},
    booktitle = {Advances In Neural Information Processing Systems 30},
    title = {{Overcoming Catastrophic Forgetting by Incremental Moment Matching (IMM)}},
    year = {2017}
    }


## Environment

Tested at

* Python v3.5
* Tensorflow v1.4.0

## Dataset

Shuffled MNIST data was used for the experiment with three sequential tasks. The first dataset is the same as the original MNIST dataset. In the second and third tasks, the input pixels of all images are shuffled with a fixed, random permutation.

no need to download mnist dataset. The download code is included. 

## Training

By default, main codes is executed with mean-imm and mode-imm. But, the main codes of SGD, L2-transfer, Drop-transfer or L2&drop-transfer are separated for its clear code.

##### Only SGD

    $ python main.py

##### L2-Transfer

    $ python main_l2.py

##### Drop-Transfer

    $ python main_drop.py

##### L2 + Drop-Transfer

    $ python main_l2_drop.py

##### with Mode IMM, Mean IMM and the other options

    $ python main_l2_drop.py --dropout 0.5 --learning_rate 0.01 --alpha 0.5 --epoch 100


##### optional arguments:
    
    --mean_imm                 # include Mean-IMM
    --mode_imm                 # include Mode-IMM

    --dropout [DROPOUT_RATE]   # dropout rate of hidden layers
    --alpha [ALPHA]            # alpha(K) of Mean & Mode IMM (cf. equation (3)~(8) in the article)
    --epoch [EPOCH]            # the number of training epoch
    --optimizer [OPTIMIZER]    # the method name of optimization. (SGD, Adam or Momentum)
    --learning_rate [RATE]     # learning rate of optimizer
    --batch_size [BATCH_SIZE]  # mini batch size
