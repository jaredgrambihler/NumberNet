import NumberNet as NN
from importData import importData
import numpy as np
import math, pickle

def main():
    """
    Basic main function to demonstrate training and running of a Network.
    Can be modified as desired for different layers, data, etc.
    """
    layers = [NN.Layer(784,512,True, 'ReLU'), NN.Layer(512,10,True)]
    #layers is a list to be used in the network. Each element is a NN.Layer object
    parameters = NN.Parameters(stepSize = 1e-4, regularization = 1e-5, decay = .9, RMSProp = False, momentum=True)
    #parameters is an object which defines the training parameters of the network.
    network = NN.Network(parameters, layers, NN.Layers.Softmax()) #initialize network with parameters, layers,
                                                                  #and softmax activation function
    trainImages, trainLabels, testImages, testLabels = importData()  #import MNIST data
    initialAccuracy = network.accuracyTest(testImages, testLabels)
    network.train(trainImages, trainLabels, testImages, testLabels, batchSize = 512, epochs = 1) #train network
    finalAccuracy = network.accuracyTest(testImages, testLabels)
    print('initAccuracy: ', initialAccuracy) #display accuracy data
    print('finalAccuracy: ', finalAccuracy)
    networkFile = open('network', 'wb') #pickle and save network for re-use
    pickle.dump(network, networkFile)
    networkFile.close()

def showData():
    """
    Runs a saved network on a given dataset and displays results
    Shows networks data from training, vizualization of weights,
    then runs the network on the given data. The displays are hardcoded
    to work with MNIST data.
    """
    networkFile = open('network', 'rb') #load network in
    network = pickle.load(networkFile)
    networkFile.close()
    network.displayData()
    network.visualizeWeights()
    trainImages, trainLabels, testImages, testLabels = importData() #load MNIST data
    network.run(testImages, testLabels, delay = 2)

main()
showData()
