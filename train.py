# encoding: utf-8
# file: main.py
# author: shawn233

from __future__ import print_function
import os
import sys
import tensorflow as tf
import numpy as np

def get_place_holder (shape, name):
    return tf.placeholder (tf.float32, shape, name=name)

if __name__ == "__main__":
    n_input = 5
    n_output = 1
    learning_rate = 0.01
    batch_size = 100
    n_epoch = 1000


    a0 = get_place_holder ([n_input, batch_size], 'a0')
    weights = {
        'layer1': tf.Variable ()
        }