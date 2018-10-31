from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(np.random.randint(1000, size=1))

import os
import sys
sys.path.insert(0, os.getcwd() + '../')
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from CreateImages import CreateImageDataset
from DataLoader import DataLoader
import tensorflow as tf

# image size
nx = 572
ny = 572
# num circles per image
numCircles = 10
# total number of training images
numImages = 1

# create the dataset
imageDS = CreateImageDataset(nx, ny, numCirlces=numCircles, numImages=numImages)

# parameters for the dataloader
params = {'batch_size': 8,
          'shuffleBuf':10,
          'r': 4,
          'epochs': 10}

# dataloader
ds = DataLoader(imageDS.filenames, params)
# define the network
net = unet.Unet(channels=imageDS.channels, n_class=imageDS.n_class, layers=3, features_root=16)
# define the trainer
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# train
path = trainer.train(ds.dataset, "./unet_trained", training_iters=numImages//params['batch_size'], epochs=params['epochs'], display_step=2)

# sample prediction
iterator = ds.dataset.make_one_shot_iterator()
x, y = iterator.get_next()

with tf.Session() as sess:
    test_x, test_y = sess.run([x, y])
    
    test_x = np.einsum('ijkl->iklj', test_x)
    test_y = np.einsum('ijkl->iklj', test_y)

    prediction = net.predict("./unet_trained/model.ckpt", test_x)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
    ax[0].imshow(test_x[0,...,0]*255, aspect="auto")
    ax[1].imshow(test_y[0,...,1], aspect="auto")
    mask = prediction[0,...,1] > 0.8
    ax[2].imshow(mask, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()
    plt.show()

