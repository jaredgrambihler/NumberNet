#Code taken from open-source Mnist Data Jupyter Notebook
#Slightly modified for use as a module

##REQUIRES python-mnist
##REQUIRES matplotlib

# files are download and loaded form a mnist folder created as a subfolder under
# the location of this python notebook
import gzip
import os
import urllib.request

mnistDir = "mnist/"

#make sure samples subfolder exists
os.makedirs(mnistDir, exist_ok=True)

# download the images from minst to a samples subfolder from this pyton notebook
# Only need to do this step once to get the data downloaded.



def downloadMnisttFile(mnistFile, outputDir):
    handle = urllib.request.urlopen("http://yann.lecun.com/exdb/mnist/" + mnistFile + ".gz") 
    zipFile = gzip.GzipFile(fileobj=handle)
    with zipFile as infile:
            with open(outputDir + mnistFile, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)

#download all the files
downloadMnisttFile("train-images-idx3-ubyte", mnistDir)
downloadMnisttFile("train-labels-idx1-ubyte", mnistDir)
downloadMnisttFile("t10k-images-idx3-ubyte", mnistDir)
downloadMnisttFile("t10k-labels-idx1-ubyte", mnistDir)


# Use the mnist module to read the data. 
# you probably have to >pip install mnist
from mnist import MNIST

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
