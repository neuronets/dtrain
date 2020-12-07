#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nobrainer
import tensorflow as tf
import os

def load_data_and_train(path, dataset_size, model, name):
    train_records = []
    eval_records = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if 'tfrec' not in filename:
                continue
            if 'train' in filename:
                train_records.append(os.path.join(path, filename))
            elif 'evaluate' in filename:
                eval_records.append(os.path.join(path, filename))
    
    print("read raw records")
    #print(str(len(train_records))+" in training set")
    #print(str(len(eval_records))+" in testing set")
    
    train_records = train_records[:dataset_size]
    
    dataset = tf.data.TFRecordDataset(train_records, compression_type="GZIP")
    print(dataset.element_spec)
    
    train_data = nobrainer.dataset.tfrecord_dataset(
        file_pattern = train_records, 
        volume_shape = (256, 256, 256),
        shuffle = None,
        scalar_label = False)
    
    
    block_shape = (128, 128, 128)
    
    train_data = train_data.map(lambda x, y: (nobrainer.volume.to_blocks(x, block_shape),
                                              nobrainer.volume.to_blocks(y, block_shape)))
    
    train_data = train_data.map(lambda x, y: (tf.reshape(x, x.shape+(1,)), tf.reshape(y, y.shape+(1,))))
    train_data = train_data.batch(256)

    #x, y = next(iter(train_data))

    num_epochs = 1

    _op = 'adam'
    _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _metrics = ['accuracy']

    model.compile(optimizer=_op, loss=_loss, metrics=_metrics)

    print("initialized and compiled models")
    for data in train_data.as_numpy_iterator():
        model.fit(data, epochs=num_epochs, verbose=2)
    model.save_weights('saved_models/'+name+'.h5', save_format = 'h5')


#example of how i use the function
num_classes = 2
shape = (128, 128, 128, 1)

model, name = (nobrainer.models.meshnet(num_classes, shape), 'meshnet')
path = os.getcwd()
size = 1
load_data_and_train(path, size, model, name)









