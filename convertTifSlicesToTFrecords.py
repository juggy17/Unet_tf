""" Examples to demonstrate how to write an image file to a TFRecord,
and how to read a TFRecord file using TFRecordReader.
"""
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io, transform
from sklearn import preprocessing
import argparse
import pandas as pd

# unit test flag
FLAGS_unitTest = True

# set verbosity
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# list of tfrecords
allFilenames = []

def normalize01(x):
    x = x.astype(np.float64)
    x -= x.mean()
    x /= x.std()
    return x

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

# module to read in the tif file with required number of zSlices into a numpy array
def ReadTifFileIntoNParrays(element, channelList, zSliceCount):
    im_out = list()
    
    # loop over the channels
    for channel in channelList:
        
        # initialize
        signalWithZslices = []
        
        # loop through the z slices
        for z in range(zSliceCount):
            #print("Processing z slice: " + str(z))
            signalFile = element[channel] + 'z' + str(z+1).zfill(2) + '.tif'  
            signal = io.imread(signalFile).astype(np.float32)
            # if signal.shape[-1] == 2:
            #     signal = signal[...,1]
            #     signal = np.expand_dims(signal, axis=-1)                
            # #signal = transform.rescale(io.imread(signalFile), scale=[0.5, 0.5], mode='reflect', multichannel=False, anti_aliasing=True).astype(np.float32)            
            signalWithZslices.append(signal)
                        
        im_out.append(np.stack(signalWithZslices, axis=0))
    
    return im_out

def write_tfrecord(element, channelList, zSliceCount):    
        
    # break it down into signal and target
    signal, target = ReadTifFileIntoNParrays(element, channelList, zSliceCount)
    
    # write to tf record
    signal = np.array(signal[0, ...]).astype(np.float32)
    target = np.array(target[0, ...]).astype(np.float32)

    feature = {'train/signal': _bytes_feature([tf.compat.as_bytes(signal.tostring())]), # _bytes_feature(signal),
               'train/target': _bytes_feature([tf.compat.as_bytes(target.tostring())]), # _bytes_feature(target),
               'train/shape':  _int64_feature(signal.shape),
               'train/tarshape': _int64_feature(target.shape)}
    
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    outname = element[channelList[0]][:-2] + '.tfrecord'    
    print("Writing: " + outname)
    allFilenames.append(outname)
    writer = tf.python_io.TFRecordWriter(outname)
    writer.write(example.SerializeToString())
    writer.close()

    return outname
        
def read_from_tfrecord(filename):
    tfrecord_file_queue = tf.train.string_input_producer(filename, name='queue')
    reader = tf.TFRecordReader()
    
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as 
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features = {"train/signal": tf.VarLenFeature(tf.string), # float32 numpy array 
                "train/target": tf.VarLenFeature(tf.string), # float32 numpy array
                "train/shape": tf.VarLenFeature(tf.int64),
                "train/tarshape": tf.VarLenFeature(tf.int64)}, name='features')
    
    # image was saved as uint8, so we have to decode as uint8.
    signal = tf.decode_raw(tf.sparse_tensor_to_dense(tfrecord_features['train/signal'], default_value=chr(0)), tf.float32)
    target = tf.decode_raw(tf.sparse_tensor_to_dense(tfrecord_features['train/target'], default_value=chr(0)), tf.float32)
    shape = tf.sparse_tensor_to_dense(tfrecord_features['train/shape'])
    tarshape = tf.sparse_tensor_to_dense(tfrecord_features['train/tarshape'])
    
    # the image tensor is flattened out, so we have to reconstruct the shape
    signal = tf.reshape(signal, shape)
    target = tf.reshape(target, tarshape)
    return signal, target, shape

# module to read tfrecord into numpy arrays
def read_tfrecord(tfrecord_file):
    print("Reading: " + tfrecord_file)
    signal, target, shape = read_from_tfrecord([tfrecord_file])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        signal, target, shape = sess.run([signal, target, shape])
        coord.request_stop()
        coord.join(threads)    

    return signal, target

# compare the original tif image with the tfrecord 
def UnitTest_CheckFidelity(signal1, target1, signal2, target2):
    # MSE
    print("Unit testing...")
    assert np.mean((signal1 - signal2)**2) == 0.0, "Test failed" 
    assert np.mean((target1 - target2)**2) == 0.0, "Test failed"

def CreateTFrecords(csvFile, zSliceCount, unitTestFlag=FLAGS_unitTest):
    # open the file for reading and look through the various tif files (along with their slices)
    assert csvFile is not None
    ds = pd.read_csv(csvFile)
    assert all(i in ds.columns for i in ['path_tif_signal', 'path_tif_target']) 
    assert zSliceCount is not None
    # list of channels per image
    channelList = ['path_tif_signal', 'path_tif_target']

    # loop over the elements of the data structure and convert them to tf records
    for index in range(ds.shape[0]):
        element = ds.iloc[index, :]
        # create tf record
        tfrecordFileName = write_tfrecord(element, channelList, zSliceCount)
        # compare input tif image and decoded tfrecord image
        if unitTestFlag:
            # read from tfrecord
            signal_tfrecord, target_tfrecord = read_tfrecord(tfrecordFileName)        
            # read from tif file
            signal_tif, target_tif = ReadTifFileIntoNParrays(element, channelList, zSliceCount)
            signal_tif = np.array(signal_tif).astype(np.float32)
            target_tif = np.array(target_tif).astype(np.float32)
            UnitTest_CheckFidelity(signal_tif, target_tif, signal_tfrecord, target_tfrecord)

    return allFilenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset_csv', default=None, type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--zSliceCount', default=None, type=int, help='Number of Z slices per tfrecord')
    opts = parser.parse_args()

    filenames = CreateTFrecords(opts.path_dataset_csv, opts.zSliceCount)
    print(filenames)

    