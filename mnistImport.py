#Code taken from open-source Mnist Data Jupyter Notebook
#Code has been simplified and edited for use as a module

##REQUIRES python-mnist
##REQUIRES matplotlib

from mnist import MNIST

mnistDir = "mnist/"

# Use the mnist module to read the data. 
mndata = MNIST(mnistDir)

#load traing data
imagesTrain, labelsTrain = mndata.load_training()

#load test data
imagesTest, labelsTest = mndata.load_testing()


# visualize the first image in the training data
# reference: https://stackoverflow.com/questions/37228371/visualize-mnist-dataset-using-opencv-or-matplotlib-pyplot
import numpy as np
import matplotlib.pyplot as plt

imageTrainIndex = 60000-1

# make a numpy array from imageTrain Python array
pixels = np.array(imagesTrain[imageTrainIndex], dtype='uint8')

# Reshape the array into 28 x 28 array (2-dimensional array)
pixels = pixels.reshape((28, 28))
    
plt.title(labelsTrain[imageTrainIndex])
plt.imshow(pixels, cmap='gray')
plt.show()

# print out and show the raw pixels
print(pixels)
