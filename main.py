import Layers as ly
import numpy as np

#class to create the network and store it in memory
class Network():

    def __init__(self):
        #init weights
        #init network
        #call output for initial accuracies
        pass

    def forwardPass(self, image):
        #run a forward pass of the network w/ an image
        #return label
        pass

    def backwardPass(self):
        #backprop through the network
        #doesn't have to return anything because
        #the weight Layer object saves it!
        pass

    def trainBatch(self, miniBatch):
        #train network on minibatch of data
        #forward/backprop through each single image
        #save all weight gradients
        #avg all gradients of the minibatch
        #update weights
        pass
    
    def accuracy(self, testData):
        #run network forwardpass on test data
        #compute % accuracy
        pass

    def outputData(self):
        #output loss
        #output accuracy
        #output weights
        pass

    def train(self, data):
        #trains data using minibatches
        #samples each minibatch
        #calls the trainBatch on it
        #outputs training time after each minibatch
        outputData()
        pass


def importData(dataSet):
    #imports dataset specified (test/train)
    #returns to main
    pass


def main():
    #init network
    numberNet = Network()
    trainData = importData('Train')
    testData = importData('Test') #import data
    initialAccuracy = numberNet.accuracy(testData) #test initial accuracy
    numberNet.train(trainData) #train network
    #save weights!
    finalAccuracy = numberNet.accuracy(testData) #test final accuracy
    #display output data
    pass


