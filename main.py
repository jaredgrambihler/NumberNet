import Layers as ly
import numpy as np
from mnist import MNIST

#class to create the network and store it in memory
class Network():

    def __init__(self, stepSize = .01):
        #init weights
        self.w1 = ly.Weights(784, 784)
        #init network
        self.l1 = ly.Multiplication()
        self.loss = ly.Softmax()
        self.stepSize = stepSize


    def forwardPass(self, image, labelIndex):
        #run a forward pass of the network w/ an image
        self.h1 = self.l1.forwardPass(self.w1, image)
        self.loss.forwardPass(self.h1, labelIndex)


    def backwardPass(self):
        #backprop through the network
        loss1 = self.loss.backwardPass()
        loss2, toss = self.l1.backwardPass(loss1) #loss2 is the loss of the weights
        return loss2


    def trainBatch(self, miniBatchImages, miniBatchLabels):
        #train network on minibatch of data
        #forward/backprop through each single image
        dW = np.zeros(784, 784) #same size as weights
        numData = len(miniBatchImages) #stores number of images being tested
        for i in range(numData): #runs through the entire miniBatch
            self.forwardPass(miniBatchImages[i], miniBatchLabels[i])
            dW += self.backwardPass() #save all weight gradients
        #avg all gradients of the minibatch
        dW = dW / numData
        #update weights
        self.w1.updateGrad(self.stepSize, dW)


    def accuracy(self, testImages, testLabels):
        accuracy = 0.0 #accuracy starts at 0, adds 1 for each correctly identified image, then takes a %
        for i in range(len(testImages)): #checks accuracy for all test data
            forwardPass(testImages[i], testLabels[i])
            largestValue = 0 #index of largest value
            for j in range(1, len(self.h1)): #finds the index of the largest score
                if (self.h1[j] > self.h1[largestValue]):
                    largestValue = j
            if (largestValue == testLabels[i]): #if the largest score is the correct class, add accuracy
                accuracy += 1           
        #compute % accuracy
        return accuracy / len(testImages)

        
    def outputData(self):
        #output loss
        #output accuracy
        #output weights
        pass


    def train(self, trainImages, trainLabels, batchSize = 25):
        #trains data using minibatches
        for i in range(len(trainImages)/batchSize):
            miniBatchImages = trainImages[i:i+batchSize] #slices train data
            miniBatchLabels = trainLabels[i:i+batchSize]
            trainBatch(miniBatchImages, miniBatchLabels)
            outputData() #some output data


#uses the python-mnist module to import the data
def importData(dir = './mnist'): #imports from a given directory, predefined to /mnist
    mndata = MNIST(dir)
    trainImages, trainLabels = mndata.load_training()
    testImages, testLabels = mndata.load_testing()
    return trainImages, trainLabels, testImages, testLabels #returns as 4 lists


def main():
    numberNet = Network() #init network
    trainImages, trainLabels, testImages, testLabels = importData() #imports data
    initialAccuracy = numberNet.accuracy(testImages, testLabels) #test initial accuracy
    numberNet.train(trainImages, trainLabels) #train network
    #pickle weights for re-use.
    finalAccuracy = numberNet.accuracy(testData) #test final accuracy
    #display output data

    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)

main()
