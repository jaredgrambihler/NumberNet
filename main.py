import NumberNet as NN
from importData import importData
import numpy as np
import math, pickle

def main():
    """
    Trains the network and pickles it after it is trained.
    Once trained, the network can be run on data without needing to be trained again.
    """
    layers = [NN.Layer(784,512,True), NN.Layer(512,512,True, 'ReLU'), NN.Layer(512,10, True), NN.Layers.Softmax()]
    parameters = NN.Parameters(stepSize = 5e-4, regularization = 1e-3, decay = .9, RMSProp = False, momentum=True)

    #init network
    network = NN.Network(parameters, layers)

    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #test initial accuracy on test data
    initialAccuracy = network.accuracyTest(testImages, testLabels)

    #trains the network
    network.train(trainImages, trainLabels, testImages, testLabels, batchSize = 512, epochs = 1)

    #tests final accuracy on test data
    finalAccuracy = network.accuracyTest(testImages, testLabels)

    #display output data
    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)

    #pickles network for re-use
    networkFile = open('network', 'wb')
    pickle.dump(network, networkFile)
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