""" Dataloader in TF using the TFRecordDtaset 
    API
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self, filenames, params):
        self.filenames = filenames
        self.params = params
        self.dataset = None
        self._data_loader()

    def _parse_function(self, example_proto):
        features = {"train/signal": tf.VarLenFeature(tf.string), # float32 numpy array 
                    "train/target": tf.VarLenFeature(tf.string), # float32 numpy array
                    "train/shape": tf.VarLenFeature(tf.int64),
                    "train/tarshape": tf.VarLenFeature(tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        full_shape = tf.sparse_tensor_to_dense(parsed_features['train/shape'])        
        full_shape.set_shape([3,])
        assert_op = tf.assert_equal(tf.shape(full_shape), np.array([3], dtype=np.int32))        

        full_tarshape = tf.sparse_tensor_to_dense(parsed_features['train/tarshape'])        
        full_tarshape.set_shape([3,])        

        signal_decoded = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['train/signal'], default_value=chr(0)), tf.float32)
        target_decoded = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['train/target'], default_value=chr(0)), tf.float32)        

        with tf.control_dependencies([assert_op]):
            signal_reshaped = tf.reshape(signal_decoded, full_shape)/255.0
            target_reshaped = tf.reshape(target_decoded, full_tarshape)/255.0
            signal_reshaped.set_shape([None, None, None])
            target_reshaped.set_shape([None, None, None])            
        
        return signal_reshaped, target_reshaped
        

    def _data_loader(self):
        # extract params
        batch_size = self.params.get('batch_size', 8)        
        shuffleBuf = self.params.get('shuffleBuf', 4)
        epochs = self.params.get('epochs', 1)
        r = self.params.get('r', 4)

        # dataset API
        self.dataset = tf.data.TFRecordDataset(self.filenames)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.shuffle(shuffleBuf)
        self.dataset = self.dataset.map(lambda e: self._parse_function(e), num_parallel_calls=4)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(2)
                
        # iterator = self.dataset.make_one_shot_iterator()
        # x, y = iterator.get_next()
        
        # count = 0

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     try: 
        #         # Keep running next_batch till the Dataset is exhausted
        #         while True:
        #             sig, tar = sess.run([x, y])
                                    
        #             plt.figure()
        #             plt.subplot(1, 2, 1)
        #             plt.imshow(sig[0,...,0])
        #             plt.subplot(1, 2, 2)
        #             plt.imshow(tar[0,...,1])
        #             plt.show(block=True)
                    
        #             count += 1
        #             print(count)
                    
        #     except tf.errors.OutOfRangeError:
        #         print('Out of range')