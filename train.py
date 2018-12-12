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
LOG_DIR = os.path.join (BASE_DIR, 'log')
SAVE_DIR = os.path.join (LOG_DIR, 'save')
MODEL_FILENAME = 'order-book-model'
OUTPUT_FILENAME = 'output.csv'

sys.path.append (UTIL_DIR)
from data_util import OrderBook




def get_model (inputs, is_training):
    '''
    Get the RNN model

    Args:
    - inputs: tf tensor;
    - is_training: tf bool tensor;

    Returns:
    - pred: prediction;
    '''

    # create RNN cell
    num_units_list = [4, 8]
    cells = [get_cell(num_units) for num_units in num_units_list]
    cell = tf.nn.rnn_cell.MultiRNNCell (cells)
    #num_units = 32
    #cell = get_cell (num_units)

    output_seq, state = tf.nn.dynamic_rnn (cell=cell, inputs=inputs, dtype=tf.float32)
    outputs = tf.reshape (output_seq, shape=[-1, num_units_list[-1] * inputs.get_shape()[1]])
    #outputs = state.h
    #print (outputs.shape)

    # additional fully connected layer
    with tf.variable_scope ('output_layer') as sc:
        weight1 = tf.get_variable ('weight1', shape=[outputs.get_shape()[-1], 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias1 = tf.get_variable ('bias1', shape=[16], dtype=tf.float32, initializer=tf.zeros_initializer())

        outputs = tf.matmul (outputs, weight1) + bias1
        #outputs = dropout (outputs, is_training, 'dropout')
        outputs = tf.nn.relu (outputs)

        weight2 = tf.get_variable ('weight2', shape=[outputs.get_shape()[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias2 = tf.get_variable ('bias2', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

        outputs = tf.matmul (outputs, weight2) + bias2
        #outputs = tf.nn.relu(tf.matmul (outputs, weight2) + bias2)

    #print (outputs.shape)
    #input()

    return outputs



def train():
    n_inputs = 10
    n_outputs = 1
    n_features = 7
    batch_size = 64
    n_epochs = 50

    inputs_pl = tf.placeholder (tf.float32, shape=[None, n_inputs, n_features], name='inputs_pl') # batch_size x len x n_features
    outputs_pl = tf.placeholder (tf.float32, shape=[None, n_outputs], name='outputs_pl') # batch_size x n_outputs
    is_training_pl = tf.placeholder (tf.bool, shape=[], name='is_training')

    pred = get_model (inputs_pl, is_training_pl)
    loss = get_loss (pred, outputs_pl)
    tf.summary.scalar ('loss', loss)

    accuracy = tf.sqrt(tf.losses.mean_squared_error (outputs_pl, pred)) # already tested
    # accuracy_my = tf.reduce_mean (tf.square (tf.subtract (outputs_pl, pred)))
    tf.summary.scalar ('accuracy', accuracy)

    step = tf.Variable (0)
    learning_rate = get_learning_rate (step, batch_size)
    tf.summary.scalar ('learning rate', learning_rate)
    train_op = tf.train.AdamOptimizer (learning_rate).minimize (loss, global_step=step)


    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    order_book = OrderBook (batch_size, DATA_DIR)
    num_batches = order_book.num_batches
    
    output_f = open (os.path.join (BASE_DIR, OUTPUT_FILENAME), 'w')
    output_f.write ('caseid,midprice\n')

    with tf.Session () as sess:
        sess.run (init)

        # create summary writers
        train_writer = tf.summary.FileWriter (os.path.join (LOG_DIR, 'train'), graph=sess.graph)
        test_writer = tf.summary.FileWriter (os.path.join (LOG_DIR, 'test'), graph=sess.graph)

        # create saver
        saver = tf.train.Saver (max_to_keep=3)

        step_val = None
        for epoch in range (n_epochs):
            order_book.reset_batch()
            total_loss = 0.0
            total_acc = 0.0
            
            for i in range (num_batches):        
                batch_inputs, batch_labels = order_book.next_batch()
                feed_dict = {inputs_pl: batch_inputs.reshape (batch_size, n_inputs, n_features), 
                            outputs_pl: batch_labels.reshape(batch_size, n_outputs),
                            is_training_pl: True}
                _, loss_val, acc_val, step_val, summary = sess.run ([train_op, loss, accuracy, step, merged],
                            feed_dict=feed_dict)
                
                # after every batch
                total_acc += acc_val
                total_loss += loss_val

                train_writer.add_summary (summary, global_step=step_val)

            print ('Epoch', epoch, 'train_loss', total_loss/num_batches, 'train_acc', total_acc/num_batches)
            dev_inputs, dev_labels = order_book.dev_set()
            feed_dict = {inputs_pl: dev_inputs.reshape (-1, n_inputs, n_features),
                        outputs_pl: dev_labels.reshape (-1, n_outputs),
                        is_training_pl: False}
            acc_val, loss_val = sess.run ([accuracy, loss], feed_dict=feed_dict)

            print ('dev_loss', loss_val, 'dev_acc', acc_val)

            saver.save (sess, os.path.join (SAVE_DIR, MODEL_FILENAME), global_step=step_val)

        '''
        test_inputs, test_labels = order_book.test_set()
        feed_dict = {inputs_pl: test_inputs.reshape (-1, n_inputs, n_features),
                    outputs_pl: test_labels.reshape (-1, n_outputs)}
        acc_val, loss_val = sess.run ([accuracy, loss], feed_dict=feed_dict)

        print ('acc', acc_val, 'loss', loss_val)
        '''

        test_data, _ = order_book.test_set()
        feed_dict = {inputs_pl:test_data, is_training_pl: False}
        pred_val = sess.run (pred, feed_dict=feed_dict)
        pred_val = np.asarray (pred_val)
        print (pred_val.shape)

        for i in range (len (pred_val)):
            if i < 142:
                continue
            output_f.write (str(i+1)+','+str(pred_val[i][0])+'\n')

    output_f.close()




def prediction ():
    '''
    Predict via restoring trained models
    '''

    pass



def get_learning_rate (
    global_step, 
    batch_size,
    base_learning_rate=1e-3,
    decay_rate=0.7,
    decay_step=200000,
    min_rate=1e-5):
    '''
    Learning rate decay by global step

    Args:
        global_step: tf variable.
        base_learning_rate: float.
        batch_size: int. 
        decay_rate: float.
        decay_step: int.
        min_rate: float. lower bound of learning rate

    Returns:
        learning_rate: tf variable.
    '''

    '''
    exponential_decay(learning_rate, global_step,
    param learning_rate
    decay_steps,                        decay_rate,
    staircase=False,                        name=None)
    '''
    learning_rate = tf.train.exponential_decay (
                    base_learning_rate,
                    global_step * batch_size,
                    decay_step,
                    decay_rate)
    learning_rate = tf.maximum(learning_rate, min_rate)
    return learning_rate 



def dropout (
    inputs,
    is_training,
    scope,
    keep_prob=0.5
):
    '''
    Dropout layer, suitable for training and testing procedure

    Args:
        inputs: tensor;
        is_training: boolean tf variable;
        scope: string;
        keep_prob: float in [0, 1].

    Returns:
        tensor variable
    '''

    with tf.variable_scope (scope) as sc:
        outputs = tf.cond (is_training,
                lambda: tf.nn.dropout (inputs, keep_prob),
                lambda: inputs)
    return outputs



def get_loss (prediction, labels):
    return tf.losses.huber_loss (labels, prediction)




def get_cell (num_units):
    '''
    Get a cell for recurrent NN

    Args:
        num_units: int, state_size
    
    Returns:
        an instance of a subclass of RNNCell 
    '''
    return tf.nn.rnn_cell.LSTMCell (num_units=num_units)




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
    train()