# Color-Segmentation
ESE 650 Learning in Robotics, Spring 2017
Project 1, Color Segmentation
Qiaoyu Zhang

Required packages:
	opencv, numpy

Tested environments:
	Windows 10, Python 3.6.0 with OpenCV 3.2.0, Numpy 1.12.0
	Ubuntu 16.04, Python 3.5.2 with OpenCV 3.2.0, Numpy 1.12.0

To test the model, run 'Predict.py'. Change test_folder on line 7 to the location of test images.

'Data Generation.py' is for selecting regions in the training images.
'Dropout.py' is for shrinking the size of the training set.
'GMM.py' is for estimating the parameters of Gaussian Mixture Models using EM algorithm
'Predict.py' is for running tests on test set.

The parameters are stored in 'ground_param.npy', 'red_barrel_param.npy', 'red_other_param.npy', 'sky_param.npy'.

