#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
#import numpy as np
import uuid
import glob
from make_model import sample_net
from file_reader import make_TFRecord_from_nii

class Client():
    def __init__(self, clientid):
        self.client_id = clientid
        shape = self.load_input_output()
        self.model = sample_net(shape)
        self.save_prior()
        
    def load_input_output(self):
        # example inputs/outputs taken from tf keras Model documentation
        # in use case, load data from files
        record_file = 'tfrecord_'+self.client_id+'.tfrec'
        shape = make_TFRecord_from_nii('data', '*_imgs*', '*_labels*', record_file)
        self.dataset = tf.data.TFRecordDataset(record_file)
        
        return shape
        #self.inputs = tf.keras.Input(shape=(3,))
        #x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(self.inputs)
        #self.outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    
    def save_prior(self):
        self.model.save_weights('prior-'+self.client_id+'.h5', save_format = 'h5')
    
    def train(self):
        _op = 'adam'
        _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        _metrics = ['accuracy']
        self.model.compile(optimizer=_op, loss=_loss, metrics=_metrics)
        #self.model.fit(inputs=self.inputs, outputs=self.outputs, epochs=1, verbose=2)
        self.model.fit(self.dataset, epochs=1, verbose=2)
    
    def load_consolidated(self):
        list_of_files = glob.glob('/server/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        self.model.load_weights(latest_file)
        return latest_file
    
    def save_weights(self):
        filename = 'server/consolidated-'+self.client_id+'-'+uuid.uuid4().__str__()+'.h5'
        self.model.save_weights(filename, save_format = 'h5')

# example usage

a = Client('A')
try:
    a.load_consolidated()
except:
    pass
a.train()
a.save_weights()
