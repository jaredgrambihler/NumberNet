import Layers as ly
import numpy as np
from mnist import MNIST
import time
import pickle
import matplotlib.pyplot as plt



#class to create the network and store it in memory. Can be pickled to avoid retraining the network.
class Network():

    #initializes the class with an optionally defined step size.
    #the layers that are stored must be written here, although
    #the interactions b/w layers are in the forward and backward pass
    #methods.
    def __init__(self, stepSize = 10e-3):
        #init weights, size is (labelNumbers, imagePixels)
        self.w1 = ly.Weights(10, 784)
        #init network layers
        #this is a single layer network w/ no activation function.
        self.l1 = ly.Multiplication()
        self.loss = ly.Softmax()
        self.stepSize = stepSize
        

    #forwards an image throught the network and outputs the loss.
    #saves the score outputs to the class.
    def forwardPass(self, image, labelIndex):
        #layer1 forward pass, happens to be only layer.
        self.h1 = self.l1.forwardPass(self.w1, image)
        #loss function forward pass.
        self.loss.forwardPass(self.h1, labelIndex)


    #backprops through the network, using values stored in the layers.
    #returns the gradient of the weight classes.
    def backwardPass(self):
        grad1 = self.loss.backwardPass()
        grad2 = self.l1.backwardPass(grad1)
        return grad2

    
    #tests the accuracy of single image, forward prop through network.
    #returns a 0.0 or a 1.0 for accuracy.
    #HAS NOT BEEN TESTED
    def accuracy(self, image, label):
        self.forwardPass(image, label)
        largestValue = 0 #index of largest value
        #finds the index of the largest score
        for j in range(1, len(self.h1)):
            #if there is a new larger value, updates index
            if (self.h1[j] > self.h1[largestValue]):
                largestValue = j
        if (largestValue == label):
            return 1
        else:
            return 0
        

    #SHOULD BE UPDATED TO WORK WITH ACCURACY
    #defines the accuracy of the network on the test data (10k images/labels)
    #returns the accuracy as a decimal value (1.0 is 100%)
    def accuracyTest(self, testImages, testLabels):
        #accuracy starts at 0, adds 1 for each correctly identified image
        accuracy = 0.0
        #loops through all test data
        for i in range(len(testImages)):
            #forward props to get scores, stored as self.h1
            self.forwardPass(testImages[i], testLabels[i])
            largestValue = 0 #index of largest value
             #finds the index of the largest score
            for j in range(1, len(self.h1)):
                #if there is a new larger value, updates index
                if (self.h1[j] > self.h1[largestValue]):
                    largestValue = j
            #if the largest score is the correct class, add accuracy
            if (largestValue == testLabels[i]):
                accuracy += 1           
        #compute % accuracy
        return accuracy / len(testImages)


    #trains the network on a minibatch of data.
    #updates the gradient after find the avg gradient for the
    #entire minibatch.
    def trainBatch(self, miniBatchImages, miniBatchLabels):
        #initializes return array the same size as weights
        dW = np.zeros(self.w1.weights.shape)
        #stores number of images being tested
        numData = len(miniBatchImages)
        #tracks accuracy, starts at 0
        accuracy = 0.0
        #tracks loss
        loss = 0.0
        #runs through the miniBatch
        for i in range(numData):
            #forwards a single image and label through the network.
            # should be done by accuarcy #self.forwardPass(miniBatchImages[i], miniBatchLabels[i])
            accuracy += self.accuracy(miniBatchImages[i], miniBatchLabels[i])
            #updates loss
            loss += self.loss.loss
            #backprops and adds to weights
            dW += self.backwardPass()
        #avg all gradients of the minibatch
        dW = dW / numData
        #update weights
        self.w1.updateGrad(self.stepSize, dW)
        #outputs data after training the minibatch
        #output loss
        loss /= numData
        self.lossList.append(loss)
        print('Loss: ', loss)
        #output accuracy
        #avg accuracy over miniBatch
        accuracy /= numData
        self.accuracyList.append(accuracy)
        print('Accuracy: ' , accuracy)
        #output weights
        weights = self.w1.weights
        self.weightsList.append(weights)
        print('Weights: ' , weights)
        #ouput changes to weights
        self.dWList.append(dW)
        print('dW: ' , dW)


    #ouputs data after a minibatch has trained
    def outputData(self):
        #output loss
        ##loss over minibatch
        #output accuracy
        ##accuracy for the minibatch
        #output weights
        ##weights = self.w1.weights
        pass


    #trains the network. Takes in train data and optional batch size.
    #outputs data on the network each minibatch.
    def train(self, trainImages, trainLabels, batchSize = 250):
        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        #Should be updated to be randomly ordered.
        numMinibatches = int(len(trainImages)/batchSize)
        #creates an index to use for slicing
        dataIndex = 0
        #times the network train time
        startTime = time.perf_counter()

        #creates output lists
        self.lossList = []
        self.accuracyList = []
        self.weightsList = []
        self.dWList = []
        
        for i in range(int(len(trainImages)/batchSize)):
            #miniBatch time tracker
            miniBatchStartTime = time.perf_counter()
            #slices train images and labels
            miniBatchImages = trainImages[dataIndex:dataIndex+batchSize]
            miniBatchLabels = trainLabels[dataIndex:dataIndex+batchSize]
            #updates dataIndex
            dataIndex += batchSize
            #trains the minibatch
            self.trainBatch(miniBatchImages, miniBatchLabels)
            #miniBatch time tracker
            miniBatchEndTime = time.perf_counter()
            #ouputs miniBatch time
            print(miniBatchEndTime - miniBatchStartTime)

        #outputs data into files
        lossFile = open('loss.txt', 'wb')
        accuracyFile = open('accuracy.txt', 'wb')
        weightsFile = open('weights.txt', 'wb')
        dWFile = open('dW.txt', 'wb')

        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.accuracyList, accuracyFile)
        pickle.dump(self.weightsList, weightsFile)
        pickle.dump(self.dWList, dWFile)
        
        #times the network train time
        endTime = time.perf_counter()
        #outputs the train time
        print('Train time: ' , endTime - startTime)



#uses the python-mnist module to import the data
#imports from a given directory, predefined to /mnist
def importData(dir = './mnist'):
    #creates mndata object from mnist data
    mndata = MNIST(dir)
    #loads train data
    trainImages, trainLabels = mndata.load_training()
    #loads test data
    testImages, testLabels = mndata.load_testing()
    #returns as 4 lists
    return trainImages, trainLabels, testImages, testLabels


#displays the data that was logged during training
#totally broken
def displayData():
    accuracy = open('accuracy.txt', 'rb')
    dW = open('dW.txt', 'rb')
    loss = open('loss.txt', 'rb')
    weights = open('weights.txt', 'rb')

    accuracyList = pickle.load(accuracy)
    lossList = pickle.load(loss)
    weightsList = pickle.load(weights)
    dWList = pickle.load(dW)

    #averages accuracy data down to n = 100 points
    avgAccuracy = []
    avgLoss =[]
##    dataPoints = 100
##    for x in range(int(len(accuracyList) / dataPoints)):
##        avgA = 0
##        avgL = 0
##        for j in range(100):
##            avgA += accuracyList[x*dataPoints + j]
##            avgL += lossList[x*dataPoints + j]
##        #appends average to the average accuracy
##        avgAccuracy.append(avgA / (len(accuracyList) / dataPoints))
##        avgLoss.append(avgL / (len(lossList) / dataPoints))

    fig, (lossPlot, accuracyPlot, weightHist0, weightHist1) = plt.subplots(1,4)

    lossPlot.plot(lossList)

    accuracyPlot.plot(accuracyList)

    #n, bins, patches = weightHist.hist(weightsList[0], 10, density = 1)
    #this code shows a histogram
    #plt.hist(data, bins = numBins)
    #plt.show()

    weights0 = weightsList[0].reshape(7840)
    weightHist0.hist(weights0, bins = 20)

    weights1 = weightsList[len(weightsList) -1].reshape(7840)
    weightHist1.hist(weights1, bins = 20)

    plt.plot(avgAccuracy)
    plt.show()


#runs network and displays labels and images 
def runNetwork():
    pass


def main():
     #init network
    numberNet = Network()
    #import data
    trainImages, trainLabels, testImages, testLabels = importData()
    #test initial accuracy
    initialAccuracy = numberNet.accuracyTest(testImages, testLabels)
    #trains the network
    numberNet.train(trainImages, trainLabels)
    #pickle weights for re-use.
    ##to be implemeneted
    #tests final accuracy
    finalAccuracy = numberNet.accuracyTest(testImages, testLabels)
    #display output data
    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)


#main()
displayData()
