import Layers as ly
import numpy as np

#class to create the network and store it in memory
class Network():

    def __init__(self, stepSize):
        #init weights
        self.w1 = ly.Weights(2,4)
        #init network
        self.l1 = ly.Multiplication()
        self.loss = ly.Softmax()
        self.stepSize = .01
        #call output for initial accuracies

    def forwardPass(self, image):
        #seperates out labels and images from an array
        if(image[1] == 'dash'): #gets labelIndex (0 or 1 for dash and block)
            labelIndex = 0
        else:
            labelIndex = 1
        image = image[0]
        #run a forward pass of the network w/ an image
        self.h1 = self.l1.forwardPass(w1, image)
        self.loss.forwardPass(self.h1, labelIndex)


    def backwardPass(self):
        #backprop through the network
        l1 = self.loss.backwardPass(1.00)
        l2, toss = self.h1.loss(l1)
        return l2

    def trainBatch(self, miniBatch):
        #train network on minibatch of data
        #forward/backprop through each single image
        dW = 0
        for x in miniBatch:
            self.forwardPass(x)
            dW += self.backwardPass() #save all weight gradients
        #avg all gradients of the minibatch
        dW = dW / len(miniBatch)
        #update weights
        self.w1.updateGrad(self.stepSize, dW)
    
    def accuracy(self, testData):
        #run network forwardpass on test data
        for x in testData:
            forwardPass(x)
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


