#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:33:42 2020

@author: Emi Z Liu
"""

# from https://github.com/corticometrics/neuroimage-tensorflow genTFrecord.py


# Creates a .tfrecord file from a directory of nifti images.
#   This assumes your niftis are soreted into subdirs by directory, and a regex
#   can be written to match a volume-filenames and label-filenames
#
# USAGE
#  python ./genTFrecord.py <data-dir> <input-vol-regex> <label-vol-regex>
# EXAMPLE:
#  python ./genTFrecord.py ./buckner40 'norm' 'aseg' buckner40.tfrecords
#
# Based off of this: 
#   http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

# imports
import tensorflow as tf
import nibabel as nib
import os, re
import numpy as np

def make_TFRecord_from_nii(data_dir, v_regex, l_regex, outfile):
    # RETURN AN INPUT SHAPE!!!
    
    def _bytes_feature(value):
    	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _int64_feature(value):
    	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def select_hipp(x):
    	x[x != 17] = 0
    	x[x == 17] = 1
    	return x
    
    def crop_brain(x):
    	x = x[90:130,90:130,90:130] #should take volume zoomed in on hippocampus area
    	return x
    
    def preproc_brain(x):
    	x = select_hipp(x)
    	x = crop_brain(x)   
    	return x
    
    def listfiles(folder):
    	for root, folders, files in os.walk(folder):
    		for filename in folders + files:
    			yield os.path.join(root, filename)
    
    def gen_filename_pairs(data_dir, v_re, l_re):
        unfiltered_filelist=list(listfiles(data_dir))
        input_list = [item for item in unfiltered_filelist if re.search(v_re,item)]
        label_list = [item for item in unfiltered_filelist if re.search(l_re,item)]
        print("input_list size:    ", len(input_list))
        print("label_list size:    ", len(label_list))
        if len(input_list) != len(label_list):
            print("input_list size and label_list size don't match")
            raise Exception
        sample_img = nib.load(input_list[0]).get_fdata()
        return (zip(input_list, label_list), sample_img)
    
    # parse args - UNCOMMENTED IN ORIGINAL CODE, NOW PASSED AS PARAMS
    # data_dir = sys.argv[1]
    # v_regex  = sys.argv[2]
    # l_regex  = sys.argv[3]
    # outfile  = sys.argv[4]
    # print("data_dir:   ", data_dir)
    # print("v_regex:    ", v_regex )
    # print("l_regex:    ", l_regex )
    # print("outfile:    ", outfile )
    
    # Generate a list of (volume_filename, label_filename) tuples
    filename_pairs, sample_img = gen_filename_pairs(data_dir, v_regex, l_regex)
    
    # To compare original to reconstructed images
    #original_images = []
    
    writer = tf.python_io.TFRecordWriter(outfile)
    for v_filename, l_filename in filename_pairs:
    
    	print("Processing:")
    	print("  volume: ", v_filename)
    	print("  label:  ", l_filename)	
    
    	# The volume, in nifti format	
    	v_nii = nib.load(v_filename)
    	# The volume, in numpy format
    	v_np = v_nii.get_data().astype('int16')
    	# The volume, in raw string format
    	v_np = crop_brain(v_np)
    	# The volume, in raw string format
    	v_raw = v_np.tostring()
    
    	# The label, in nifti format
    	l_nii = nib.load(l_filename)
    	# The label, in numpy format
    	l_np = l_nii.get_data().astype('int16')
    	# Preprocess the volume
    	l_np = preproc_brain(l_np)
    	# The label, in raw string format
    	l_raw = l_np.tostring()
    
    	# Dimensions
    	x_dim = v_np.shape[0]
    	y_dim = v_np.shape[1]
    	z_dim = v_np.shape[2]
    	print("DIMS: " + str(x_dim) + str(y_dim) + str(z_dim))
    
    	data_point = tf.train.Example(features=tf.train.Features(feature={
    		'image_raw': _bytes_feature(v_raw),
    		'label_raw': _bytes_feature(l_raw)}))
        
    	writer.write(data_point.SerializeToString())
    
    writer.close()
    return sample_img.shape