# encoding: utf-8
# file: main.py
# author: shawn233

from __future__ import print_function
import os
import sys
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname (os.path.abspath (sys.argv[0]))
DATA_DIR = os.path.join (BASE_DIR, 'data')
UTIL_DIR = os.path.join (BASE_DIR, 'util')

sys.path.append (UTIL_DIR)
from data_util import OrderBook

def get_cell (num_units):
    '''
    Get a cell for recurrent NN

    Args:
        num_units: int, state_size
    
    Returns:
        an instance of a subclass of RNNCell 
    '''
    return tf.nn.rnn_cell.BasicLSTMCell (num_units=num_units)

def train():
    n_features = 7
    n_input_sequence = 10
    n_output_sequence = 20

    inputs_pl = tf.placeholder (tf.float32, shape=[None, n_input_sequence, n_features]) # batch_size x input_sequence x n_features
    labels_pl = tf.placeholder (tf.float32, shape=[None, n_output_sequence]) # batch_size x output_sequence
    zeros = np.

    # define rnn cells
    n_layers = 3
    cells = [get_cell(64), get_cell(256), get_cell(128)]
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    # use an auto-encoder style
    _, state = tf.nn.dynamic_rnn (cell=rnn_cell, inputs=inputs_pl, dtype=tf.float32, time_major=False)
    outputs, _ = tf.nn.dynamic_rnn (cell=rnn_cell, inputs, dtype=tf.float32, time_major=False)


def test():
    dataset = [[[1], [2], [3], [4]], 
                [[2], [4], [1]], 
                [[5], [2], [7], [3], [8]],
                [[9], [7]]] 
    n_features = 1
    n_samples = len (dataset)
    lengths = [len(dataset[i]) for i in range(n_samples)]

    max_length = max (lengths)

    padding_dataset = np.zeros ([n_samples, max_length, n_features])
    for idx, seq in enumerate (dataset):
        padding_dataset[idx, :len(seq), :] = seq
    inputs = tf.constant (padding_dataset, dtype=tf.float32)
    print (inputs.get_shape())

    #print (padding_dataset)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell (num_units=64)
    outputs, state = tf.nn.dynamic_rnn (cell=lstm_cell, inputs=inputs, sequence_length=lengths, dtype=tf.float32)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        res = sess.run (outputs)
        print (res.shape)


if __name__ == "__main__":
    test()