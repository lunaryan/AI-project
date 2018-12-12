# encoding: utf-8
# file: data_util.py
# author: shawn233

from __future__ import print_function
import os
import sys
import numpy as np
import joblib

'''
Functions:
1. divide data into training set, dev set and test set (7:1:2)
2. provide function `next_batch()`, returns the next batch in one epoch;
   provide function `reset_batch()`, to reset the batch for a new epoch.

Usage tips:
1. Assume memory is large enough to store all data, will use `readlines()` to read data;
'''

TRAIN_INPUTS_FILENAME = 'train_inputs.npy'
TRAIN_LABELS_FILENAME = 'train_labels.npy'
DEV_INPUTS_FILENAME = 'dev_inputs.npy'
DEV_LABELS_FILENAME = 'dev_labels.npy'
TEST_INPUTS_FILENAME = 'test_inputs.npy'
TEST_LABELS_FILENAME = 'test_labels.npy'

TRAIN_DATA_FILENAME = 'train_data.csv'
TEST_DATA_FILENAME = 'test_data.csv'



def _save_data (inputs, labels, full_path_dir, inputs_name, label_name):
    '''
    Save data into full_path_dir
    
    used in function `divide_data()`
    '''

    #arr_inputs = np.array(inputs, dtype=np.float32)
    #arr_labels = np.array(labels, dtype=np.float32)

    np.save(os.path.join (full_path_dir, inputs_name), inputs)
    np.save(os.path.join (full_path_dir, label_name), labels)


def _read_data (full_path_dir, inputs_name, labels_name):
    '''
    Read data from full_path_dir
    
    Returns:
        inputs, labels
    '''

    return np.load (os.path.join (full_path_dir, inputs_name)),\
            np.load (os.path.join (full_path_dir, labels_name))


class OrderBook:

    '''
    Order book class, designed mainly for data input
    '''

    def __init__ (self, batch_size, data_dir, num_inputs=10, num_labels=20, num_features=7):
        '''
        Initialization, open the files and set the arguments

        Args:
            batch_size: int;
            data_dir: string, directory of the data
        '''

        self._batch_size = batch_size
        self.batch_ind = 0
        self._data_dir = data_dir
        self._num_inputs = num_inputs
        self._num_labels = num_labels
        self._num_features = num_features

        # vars for training set
        self.train_inputs = None
        self.train_labels = None

        # vars for dev set
        self.dev_inputs = None
        self.dev_labels = None

        # vars for test set
        self.test_inputs = None
        self.test_labels = None

        if not os.path.exists (os.path.join (self.data_dir, TRAIN_INPUTS_FILENAME)):
            # a weak test
            self.__divide_data(os.path.join (self.data_dir, TRAIN_DATA_FILENAME))
            self.__read_test_data (os.path.join (self.data_dir, TEST_DATA_FILENAME))


    @property
    def batch_size (self):
        return self._batch_size

    
    @batch_size.setter
    def batch_size (self, value):
        self._batch_size = value

    
    @property
    def data_dir (self):
        return self._data_dir


    @data_dir.setter
    def data_dir (self, value):
        self._data_dir = value


    @property
    def num_samples (self):
        '''
        Number of training samples
        '''
        if self.train_labels is None:
            self.train_inputs, self.train_labels = _read_data (self.data_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)

        return len (self.train_labels)

    
    @property
    def num_batches (self):
        '''
        Maximum number of batches that can be provided in one epoch
        '''
        return int (self.num_samples / self.batch_size)


    def __divide_data (self, full_path_filename):
        '''
        Divide data into training set, dev set and test set (7:1:2)
        
        Args:
            full_path_filename: string, full path of the original data file
        
        Returns:
            None

        (Implementation specified for projects, can not be reused)
        '''
        input_f = open (full_path_filename, 'r')
        input_size = 10
        output_avg_len = 20

        # 1. read data
        prev_date = None
        inputs = []
        labels = []
        line_cnt = 0 # cnt the lines already read

        day_entries = None # all entries in a day

        input_f.readline() # skip the first line

        for raw_line in input_f:
            
            # 1.1 read through a day
            line = raw_line.strip().split(',')
            date = line[1]

            if (prev_date is None) or prev_date != date:
                
                if (prev_date is not None):
                    # means data for date are collected
                    num_inputs = (len(day_entries) - output_avg_len) // input_size
                    for i in range (num_inputs):
                        inputs.append (day_entries[input_size * i: input_size * (i+1)])
                        
                        avg_mid_price = 0.0
                        for j in range (output_avg_len):
                            avg_mid_price += day_entries[input_size * (i+1) + j][0]
                        avg_mid_price /= output_avg_len

                        labels.append (avg_mid_price)

                prev_date = date
                day_entries = []

            # 1.2 preprocess data: type + normalization
            for i in range (3, len(line)):
                line[i] = float (line[i])
            self.__normalize (line)

            day_entries.append (line[3:])

            line_cnt += 1
            if line_cnt % 5000 == 0:
                print ('line', line_cnt, 'finished')

            
        # 2. after all inputs and labels are stored, shuffle indices
        length = len (inputs)
        indices = list (range(length))
        np.random.shuffle (indices)

        train_data_bound = int (0.8 * length)
        dev_data_bound = int (length)
        #test_data_bound = int (length) # not used, just for demonstration

        print ('# train', train_data_bound)
        print ('# dev', dev_data_bound - train_data_bound)
        #print ('# test', test_data_bound - dev_data_bound)


        # 3. save divided data respectively
        train_inputs = []
        train_labels = []
        for ind in indices[:train_data_bound]:
            train_inputs.append (inputs[ind])
            train_labels.append (labels[ind])
        
        dev_inputs = []
        dev_labels = []
        for ind in indices[train_data_bound:]:
            dev_inputs.append (inputs[ind])
            dev_labels.append (labels[ind])

        '''
        test_inputs = []
        test_labels = []
        for ind in indices[dev_data_bound:]:
            test_inputs.append (inputs[ind])
            test_labels.append (labels[ind])
        '''

        full_path_dir = os.path.dirname (full_path_filename)
        _save_data (train_inputs, train_labels, full_path_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)
        _save_data (dev_inputs, dev_labels, full_path_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)
        #_save_data (test_inputs, test_labels, full_path_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)


    
    def __read_test_data (self, full_path_filename):
        '''
        Read and save test data set
        '''

        f = open (full_path_filename, 'r')

        f.readline() # skip the first line

        line_cnt = 0
        case_line_cnt = 0
        n_case_inputs = 10

        inputs = []
        case_inputs = []

        for raw_line in f:
            if line_cnt % 5000 == 0 and line_cnt != 0:
                print ('line', line_cnt, 'finished')
            line_cnt += 1
            line = raw_line.strip()
            
            if line == '':
                # separate line
                assert len (case_inputs) == n_case_inputs 
                inputs.append (case_inputs)
                case_inputs = []
                case_line_cnt = 0
                continue

            line = line.split (',')
            case_line_cnt += 1
            
            for i in range (3, len(line)):
                line[i] = float(line[i])
            self.__normalize (line)
            case_inputs.append (line[3:])

        f.close()

        full_path_dir = os.path.dirname (full_path_filename)
        _save_data (inputs, [], full_path_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)


    def __normalize (self, line):
        '''
        Make large terms smaller
        '''
        line[5] = line[5]/1e8
        line[7] = line[7]/1e6
        line[9] = line[9]/1e6


    def __padding (self, inputs, labels):
        '''
        Pad inputs and labels into trainable shape

        Args:
            inputs: a list of inputs
            labels: a list of labels

        Returns:
            train_inputs: a padded array of inputs
            train_labels: a padded array of labels
            train_lengths: a list of ints (actual lengths)
        '''

        train_lengths = [len(e) for e in labels]
        max_length = max (train_lengths)

        train_inputs = np.zeros ([len(inputs), max_length, len (inputs[0][0])], dtype=np.float32)
        train_labels = np.zeros ([len(labels), max_length], dtype=np.float32)

        for ind in range(len(inputs)):
            assert len (inputs[ind]) == len (labels[ind])
        
            train_inputs[ind, :len(inputs[ind]), :] = inputs[ind]
            train_labels[ind, :len(labels[ind])] = labels[ind]

        return train_inputs, train_labels, train_lengths



    def next_batch (self):
        '''
        Get the next batch of the training set

        Returns:
            train_inputs_batch: a padded input batch, batch_size x max_len x n_input
            train_labels_batch: a padded label batch, batch_size x 1
        '''

        if self.train_inputs is None:
            self.train_inputs, self.train_labels = _read_data (self.data_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)

        assert self.batch_ind + self.batch_size <= len (self.train_inputs)

        train_batch_inputs = self.train_inputs[self.batch_ind: self.batch_ind + self.batch_size]
        train_batch_labels = self.train_labels[self.batch_ind: self.batch_ind + self.batch_size]

        self.batch_ind += self.batch_size

        return train_batch_inputs, train_batch_labels


    def reset_batch (self):
        '''
        Reset self.batch_ind for a new epoch
        '''

        self.batch_ind = 0


    def dev_set (self):    
        '''
        Get the padded dev inputs and labels

        Returns:
            dev_inputs: a list of inputs (lists);
            dev_lables: a list of labels
        '''

        if self.dev_inputs is None:
            self.dev_inputs, self.dev_labels = _read_data (self.data_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)

        return self.dev_inputs, self.dev_labels


    def test_set (self):
        '''
        Get the padded test inputs and labels

        Returns:
            test_inputs: a list of inputs
            test_labels: a list of labels
        '''

        if self.test_inputs is None:
            self.test_inputs, self.test_labels = _read_data (self.data_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)

        return self.test_inputs, self.test_labels





if __name__ == "__main__":
    BASE_DIR = os.path.dirname (os.path.abspath(sys.argv[0]))
    #INPUT_FILENAME = 'train1.csv'
    PROJECT_DIR = os.path.dirname (BASE_DIR)
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')

    order_book = OrderBook (256, DATA_DIR)
    print (order_book.num_batches)
    for i in range (10):
        inputs, labels = order_book.next_batch ()
        print (inputs.shape)
        print (labels.shape)

    test_inputs, _ = order_book.test_set()
    print (test_inputs)

