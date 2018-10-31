from __future__ import division, print_function
from tf_unet import image_gen
from tf_unet import util
import numpy as np
import pandas as pd
import os
import sys
import convertTifSlicesToTFrecords
import matplotlib.pyplot as plt

SAVE_DIR = 'data/'

class CreateImageDataset(object):
    
    def __init__(self, nx, ny, numCirlces=20, numImages=100):
        
        self.numImages = numImages

        # define the generator
        self.generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=numCirlces)
        
        # number of channels and output classes
        self.channels = self.generator.channels
        self.n_class = self.generator.n_class

        # generate the image set
        self._gen_imageset()
    
    def _gen_imageset(self):
        
        # generate images and their ground truths
        x, y = self.generator(self.numImages)
        
        # dataframe info
        dictInfo = {'path_tif_signal': [], 'path_tif_target': []}
        
        # save images set as tif
        for imageIdx in range(x.shape[0]):
            signal = x[imageIdx, ...]
            target = y[imageIdx, ...]
            
            print('Generating image {:d}'.format(imageIdx + 1))
            
            signal_rot = np.einsum('ijk->kij', signal*255)
            target_rot = np.einsum('ijk->kij', target*255)
            util.saveImageAsTiff(signal_rot, SAVE_DIR + 'signal_{:s}c2z01.tif'.format(str(imageIdx).zfill(2)))
            util.saveImageAsTiff(target_rot, SAVE_DIR + 'target_{:s}c1z01.tif'.format(str(imageIdx).zfill(2)))

            # append to dict
            dictInfo['path_tif_signal'].append(SAVE_DIR + 'signal_{:s}c2'.format(str(imageIdx).zfill(2)))
            dictInfo['path_tif_target'].append(SAVE_DIR + 'target_{:s}c1'.format(str(imageIdx).zfill(2)))

        # convert to pandas dataframe and write to csv
        df = pd.DataFrame(dictInfo, columns=['path_tif_signal', 'path_tif_target'])
        df.to_csv('tifData.csv', index=False)

        # convert the tif files to tfrecords and get the list of filenames
        self.filenames = convertTifSlicesToTFrecords.CreateTFrecords('tifData.csv', 1, False)
        
