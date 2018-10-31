=============================
Tensorflow Unet
=============================

.. image:: https://readthedocs.org/projects/tf-unet/badge/?version=latest
	:target: http://tf-unet.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status
	
.. image:: http://img.shields.io/badge/arXiv-1609.09077-orange.svg?style=flat
        :target: http://arxiv.org/abs/1609.09077

.. image:: https://raw.githubusercontent.com/jakeret/tf_unet/master/docs/toy_problem.png
   :alt: Segmentation of a toy problem.
   :align: center
		

This is a generic **U-Net** implementation as proposed by `Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_ developed with **Tensorflow**. 

The goal of this repository is to act as a template to be able to do the following - 

1. Generate a sample dataset consisting of circular and rectangular regions in an image that need to be segmented out.
2. Save the sample dataset in the form of layered TIF files.
3. Convert the TIF files into TFRecords
4. Use the TensorFlow Dataset API to create a dataloader based on the TFRecords and train the U-net to segment out the desired regions.

Usage:

1. Install Anaconda
2. Create a virtual environment with Python 3.6 and install all packages in the "Requirements" document. (Anaconda cheat sheet: https://conda.io/docs/_downloads/conda-cheatsheet.pdf)
3. Run: python toyProblem.py
4. Parameters like image size, number of images, number of epochs etc. are all set in toyProblem.py


As you use **tf_unet** for your exciting discoveries, please cite the paper that describes the package::


	@article{akeret2017radio,
	  title={Radio frequency interference mitigation using deep convolutional neural networks},
	  author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre},
	  journal={Astronomy and Computing},
	  volume={18},
	  pages={35--39},
	  year={2017},
	  publisher={Elsevier}
	}
