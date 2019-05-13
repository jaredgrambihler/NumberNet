import Layers as ly
import numpy as np
from mnist import MNIST
import time
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def importData(dir = './mnist'):
    """
    Returns data of MNIST dataset.
    Uses the python-mnist module to import this data.
    If the directory of your data is different than /mnist,
    this method call can be edited.
    """
    try:
        #creates mndata object from mnist data
        mndata = MNIST(dir)

        #loads train data
        trainImages, trainLabels = mndata.load_training()

        #loads test data
        testImages, testLabels = mndata.load_testing()

        #returns as 4 lists
        return trainImages, trainLabels, testImages, testLabels
    except FileNotFoundError:
        print('Need to get MNIST data or change directory.')
        return None
    return None

def main():
    """
    Trains the network and pickles it after it is trained.
    Once trained, the network can be run on data without needing to be trained again.
    """

    #layer = (Weights, Multiplication, Bias, Activation),...(loss)

    #layers = [createLayer(784, 10, True), ly.Softmax()]
    layers = [createLayer(784,512,True), createLayer(512,512,True, 'ReLU'), createLayer(512,10, True), ly.Softmax()]
    parameters = Parameters(stepSize = 5e-4, regularization = 1e-3, decay = .9, RMSProp = False, momentum=True)

    #init network
    numberNet = Network(parameters, layers)

    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #test initial accuracy on test data
    initialAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #trains the network
    numberNet.train(trainImages, trainLabels, testImages, testLabels, parameters, batchSize = 256, epochs = 10)
    #numberNet.train(trainImages[:10], trainLabels[:10], testImages, testLabels, parameters, batchSize = 10, epochs = 1000)


    #tests final accuracy on test data
    finalAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #display output data
    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)

    #pickles network for re-use
    networkFile = open('network', 'wb')
    pickle.dump(numberNet, networkFile)
    networkFile.close()

def showData():
    networkFile = open('network', 'rb')
    network = pickle.load(networkFile)
    networkFile.close()

    network.displayData()

    network.visualizeWeights()
    
    trainImages, trainLabels, testImages, testLabels = importData()
    network.run(testImages, testLabels, delay = 2)

main()
showData()