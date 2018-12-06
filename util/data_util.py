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

DATA_FILENAME = 'train1.csv'




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
            self.__divide_data(os.path.join (self.data_dir, DATA_FILENAME))


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

        # 1. read data
        data = [] # ordered by date, each element is a dict, 
                # {'date':*, 'midprice':*, 'lastprice':*, 'volume':*, \
                # 'bidprice1':*, 'bidvolume1':*, 'askprice1':*, 'askvolume1':*}
        prev_date = None
        record = None

        key_list = ['midprice', 'lastprice', 'volume', 'bidprice1', 'bidvolume1', 'askprice1', 'askvolume1']

        input_f.readline() # skip the first line
        cnt = 0
        for raw_line in input_f:
            line = raw_line.strip().split(',')
            date = line[1].split(' ')[0]
            if (prev_date is None) or (date != prev_date):
                if prev_date:
                    data.append (record)

                # a new record
                record = {
                    'date':date,
                    'midprice':[],
                    'lastprice':[],
                    'volume':[],
                    'bidprice1':[],
                    'bidvolume1':[],
                    'askprice1':[],
                    'askvolume1':[]
                }
                
                prev_date = date

            record['midprice'].append (float(line[2]))
            record['lastprice'].append (float(line[3]))
            record['volume'].append (int(line[4]))
            record['bidprice1'].append (float(line[5]))
            record['bidvolume1'].append (int(line[6]))
            record['askprice1'].append (float (line[7]))
            record['askvolume1'].append (int (line[8]))

            cnt += 1
            if cnt % 1000 == 0:
                print ('line', cnt, 'finished')
        
        assert data[-1]['date'] != record['date']
        data.append (record)

        input_f.close()

        # 2. process data to inputs and labels

        inputs = []
        labels = []
        # define three constants here, specified by the project
        NUM_INPUTS = self._num_inputs
        NUM_LABELS = self._num_labels
        NUM_FEATURES = self._num_features

        for record in data:
            # each record corresponds to a date
            # number of inputs
            num = int((len (record['midprice']) - NUM_LABELS) / NUM_INPUTS)
            for ind in range(num):
                element_inputs = np.zeros([NUM_INPUTS, NUM_FEATURES], dtype=np.float32)
                element_labels = np.zeros([NUM_LABELS], dtype=np.float32)
                start_ind = ind * NUM_INPUTS
                assert start_ind + NUM_LABELS <= len (record['midprice'])
                
                for i in range (len(key_list)):
                    element_inputs[:, i] = record[key_list[i]][start_ind:start_ind+NUM_INPUTS]
                element_labels = record['midprice'][start_ind:start_ind+NUM_LABELS]
                
                inputs.append (element_inputs)
                labels.append (element_labels)

        print ('# samples', len(inputs))

        # 3. shuffle indices

        length = len(inputs)
        indices = list (range(length))
        np.random.shuffle (indices)
        
        train_data_bound = int (0.7 * length)
        dev_data_bound = int (0.8 * length)
        test_data_bound = int (length) # not used, just for demonstration

        print ('# train', train_data_bound)
        print ('# dev', dev_data_bound - train_data_bound)
        print ('# test', test_data_bound - dev_data_bound)

        # 4. save divided data respectively
        train_inputs = []
        train_labels = []
        for ind in indices[:train_data_bound]:
            train_inputs.append (inputs[ind])
            train_labels.append (labels[ind])
        
        dev_inputs = []
        dev_labels = []
        for ind in indices[train_data_bound:dev_data_bound]:
            dev_inputs.append (inputs[ind])
            dev_labels.append (labels[ind])

        test_inputs = []
        test_labels = []
        for ind in indices[dev_data_bound:]:
            test_inputs.append (inputs[ind])
            test_labels.append (labels[ind])

        full_path_dir = os.path.dirname (full_path_filename)
        _save_data (train_inputs, train_labels, full_path_dir, TRAIN_INPUTS_FILENAME, TRAIN_LABELS_FILENAME)
        _save_data (dev_inputs, dev_labels, full_path_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)
        _save_data (test_inputs, test_labels, full_path_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)


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
            train_labels_batch: a padded label batch, batch_size x max_len x 1
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
            dev_lables: a list of labels (lists)
        '''

        if not self.dev_inputs:
            self.dev_inputs, self.dev_labels = _read_data (self.data_dir, DEV_INPUTS_FILENAME, DEV_LABELS_FILENAME)

        return self.dev_inputs, self.dev_labels


    def test_set (self):
        '''
        Get the padded test inputs and labels

        Returns:
            test_inputs: a list of inputs
            test_labels: a list of labels
        '''

        if not self.test_inputs:
            self.test_inputs, self.test_labels = _read_data (self.data_dir, TEST_INPUTS_FILENAME, TEST_LABELS_FILENAME)

        return self.test_inputs, self.test_labels





if __name__ == "__main__":
    BASE_DIR = os.path.dirname (os.path.abspath(sys.argv[0]))
    #INPUT_FILENAME = 'train1.csv'
    PROJECT_DIR = os.path.dirname (BASE_DIR)
    DATA_DIR = os.path.join (PROJECT_DIR, 'data')

    order_book = OrderBook (256, DATA_DIR)
    for i in range (order_book.num_batches):
        inputs, labels = order_book.next_batch ()
        print (inputs.shape)
        print (labels.shape)

    order_book.dev_set()
    order_book.test_set()

