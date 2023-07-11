# Wilt

### About
Detecting diseased trees in QuickBird imagery using Neural Networks.

In this implementation the Wilt dataset can been considered to classify multispectal remote sensing images into diseased tree and other land form images. 

### Dataset
This implementation uses he wilt dataset from the UCI Machine Learning Repository. It consists of 4339 image segments for training and 500 image segments for testing. 

##### Attributes
* Class: 'w' (diseased trees), 'n' (all other land cover) 
* GLCM_Pan: GLCM mean texture (Pan band) 
* Mean_G: Mean green value
* Mean_R: Mean red value
* Mean_NIR: Mean NIR value 
* SD_Pan: Standard deviation (Pan band) 

The dataset can be found in [here](http://archive.ics.uci.edu/ml/datasets/wilt)

### Implementation
Python and its various toolkits have been used in the classification of diseased tree images using neural networks.
* Keras - a high-level neural networks API, capable of running on top of TensorFlow.
* Numpy - a fundamental package for scientific computing with Python.
* Scikit-learn - a machine learning library for Python.

### Performance
For pre-determined training and testing datasets.
Performance : 99.35 % ( for 100 epochs)

### Data files 
* testing.csv: 500 samples, used for testing
* training.csv: 4339 samples, used for training


