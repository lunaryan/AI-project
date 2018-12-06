# encoding: utf-8
# file: main.py
# author: shawn233

from __future__ import print_function
import os
import sys
import tensorflow as tf
import numpy as np

def train():
    N_INPUT = 5
    BATCH_SIZE = 50
    LEARNING_RATE = 0.01
    NUM_UNITS = 128
    
    input_pl = tf.placeholder (tf.float32, shape=[BATCH_SIZE, None, N_INPUT]) # batch_size x max_time x n_input
    label_pl = tf.placeholder (tf.float32, shape=[BATCH_SIZE, None, 1]) # batch_size x max_time x n_input

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell (num_units=NUM_UNITS)

    outputs, state = tf.nn.dynamic_rnn (lstm_cell, input_pl)

def test():
    dataset = [[[1], [2], [3], [4]], 
                [[2], [4], [1]], 
                [[5], [2], [7], [3], [8]],
                [[9], [7]]] 
    n_features = 1
    n_samples = len (dataset)
    lengths = [len(dataset[i]) for i in range(n_samples)]

    max_length = max (lengths)

    padding_dataset = np.zeros ([n_samples, max_length, n_features], dtype=np.float32)

    for idx, seq in enumerate (dataset):
        padding_dataset[idx, :len(seq), :] = seq

    #print (padding_dataset)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell (64)
    outputs, state = tf.nn.dynamic_rnn (cell=lstm_cell, inputs=padding_dataset, sequence_length=lengths, dtype=tf.float32)

    with tf.Session() as sess:
        res = sess.run (outputs)
        print (res)


if __name__ == "__main__":
    test()