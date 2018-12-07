import Layers as ly
import numpy as np
from mnist import MNIST
import time
import pickle
import math
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
        #init Biases
        self.b1 = ly.Bias(10)
        #init network layers
        #this is a single layer network w/ no activation function.
        self.l1 = ly.Multiplication()
        self.loss = ly.Softmax()
        self.stepSize = stepSize
        

    #forwards an image throught the network.
    #saves the score outputs and loss to the class.
    def forwardPass(self, image, labelIndex):
        #layer1 forward pass, happens to be only layer.
        self.h1 = self.l1.forwardPass(self.w1, image)
        self.h2 = self.b1.forwardPass(self.h1)
        #loss function forward pass.
        self.loss.forwardPass(self.h2, labelIndex)


    #backprops through the network, using values stored in the layers.
    #returns the gradient of the weight classes.
    def backwardPass(self):
        grad = self.loss.backwardPass()
        biasGrad = self.b1.backwardPass(grad)
        weightGrad, imageGrad = self.l1.backwardPass(biasGrad)
        return biasGrad, weightGrad

    
    #tests the accuracy of single image, forward prop through network.
    #returns a 0 or 1 for accuracy.
    def accuracy(self, image, label):
        self.forwardPass(image, label)
        largestValue = 0 #index of largest value
        #finds the index of the largest score
        for j in range(1, len(self.h1)):
            #if there is a new larger value, updates index
            if (self.h2[j] > self.h2[largestValue]):
                largestValue = j
        if (largestValue == label):
            return 1
        else:
            return 0
        
    
    #UPDATED TO WORK WITH ACCURACY
    #HAS NOT BEEN TESTED
    #defines the accuracy of the network on the test data (10k images/labels)
    #returns the accuracy as a decimal value (1.0 is 100%)
    def accuracyTest(self, testImages, testLabels):
        #accuracy starts at 0, adds 1 for each correctly identified image
        accuracy = 0.0
        ##loops through all test data
        for i in range(len(testImages)):
        #    #forward props to get scores, stored as self.h1
        #    self.forwardPass(testImages[i], testLabels[i])
        #    largestValue = 0 #index of largest value
        #     #finds the index of the largest score
        #    for j in range(1, len(self.h1)):
        #        #if there is a new larger value, updates index
        #        if (self.h1[j] > self.h1[largestValue]):
        #            largestValue = j
        #    #if the largest score is the correct class, add accuracy
        #    if (largestValue == testLabels[i]):
        #        accuracy += 1
            accuracy += self.accuracy(testImages[i], testLabels[i])           
        
        #compute % accuracy
        return accuracy / len(testImages)


    #trains the network on a minibatch of data.
    #updates the gradient after find the avg gradient for the
    #entire minibatch.
    def trainBatch(self, miniBatchImages, miniBatchLabels):
        #initializes return array the same size as weights
        dW = np.zeros(self.w1.weights.shape)
        dB = np.zeros(self.b1.bias.shape)
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
            biasGrad, weightGrad = self.backwardPass()
            dW += weightGrad
            dB += biasGrad
        #avg all gradients of the minibatch
        dW = dW / numData
        dB = dB / numData
        #update weights
        self.w1.updateGrad(self.stepSize, dW)
        self.b1.updateGrad(self.stepSize, dB)
        #outputs data after training the minibatch
        #should be upated to be used w/ outputData
        #output loss
        loss = loss / numData
        self.lossList.append(loss)
        print('Loss: ', loss)
        #output accuracy
        #avg accuracy over miniBatch
        accuracy = accuracy / numData
        self.accuracyList.append(accuracy)
        print('Accuracy: ' , accuracy)
        #output weights
        weights = self.w1.weights
        self.weightsList.append(weights)
        print('Weights: ' , weights)


    #ouputs data after a minibatch has trained
    def outputData(self, loss, accuracy):
        #output loss
        self.lossList.append(loss)
        print('Loss: ', loss)
        #output accuracy
        #avg accuracy over miniBatch
        self.accuracyList.append(accuracy)
        print('Accuracy: ' , accuracy)
        #output weights
        weights = self.w1.weights
        self.weightsList.append(weights)
        print('Weights: ' , weights)

      
    #trains the network. Takes in train data and optional batch size.
    #outputs data on the network each minibatch.
    def train(self, trainImages, trainLabels, batchSize = 500):
        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        #Should be updated to be randomly ordered and have iterations
        numMinibatches = int(len(trainImages)/batchSize)
        #creates an index to use for slicing
        dataIndex = 0
        #times the network train time
        startTime = time.perf_counter()

        #creates output lists
        self.lossList = []
        self.accuracyList = []
        self.weightsList = []

        #opens loss files
        lossFile = open('loss', 'wb')
        accuracyFile = open('accuracy', 'wb')
        weightsFile = open('weights', 'wb')

        self.weightsList.append(self.w1.weights)
        
        #trains batches
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
        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.accuracyList, accuracyFile)
        pickle.dump(self.weightsList, weightsFile)
        

        lossFile.close()
        accuracyFile.close()
        weightsFile.close()

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
    accuracy = open('accuracy', 'rb')
    loss = open('loss', 'rb')
    weights = open('weights', 'rb')

    accuracyList = pickle.load(accuracy)
    lossList = pickle.load(loss)
    weightsList = pickle.load(weights)

    accuracy.close()
    loss.close()
    weights.close()

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


    plt.show()


#needs to be updated to show all weights at once
def visualizeWeights():
    #load weights
    weights = open('weights', 'rb')
    weightsList = pickle.load(weights)
    weights.close()

    #makes a list of np arrays for all weights
    weights = []
    for i in range(10):
        weights.append(np.array (weightsList[len(weightsList)-1][i]) )
        #weights.append(np.array (weightsList[0][i]) )

    #finds max to scale images
    maxPixel = 0
    #for each set of weights
    for weight in weights:
        #for each element in the weights
        for element in weight:
            #finds max value in ALL weights (shows importance among all scores equally)
            if (abs(element) > maxPixel):
                maxPixel = abs(element)

    #scales array
    scalar = 255.0 / maxPixel
    for i in range(len(weights)):
        weights[i] *= scalar
        weights[i] = np.reshape(weights[i], (28,28))

    #plots
    fig = plt.figure()
    ax1 = fig.add_subplot(251)
    ax1.imshow(weights[1], cmap = 'gray')
    ax2 = fig.add_subplot(252)
    ax2.imshow(weights[2], cmap = 'gray')
    ax3 = fig.add_subplot(253)
    ax3.imshow(weights[3], cmap = 'gray')
    ax4 = fig.add_subplot(254)
    ax4.imshow(weights[4], cmap = 'gray')
    ax5 = fig.add_subplot(255)
    ax5.imshow(weights[5], cmap = 'gray')
    ax6 = fig.add_subplot(256)
    ax6.imshow(weights[6], cmap = 'gray')
    ax7 = fig.add_subplot(257)
    ax7.imshow(weights[7], cmap = 'gray')
    ax8 = fig.add_subplot(258)
    ax8.imshow(weights[8], cmap = 'gray')
    ax9 = fig.add_subplot(259)
    ax9.imshow(weights[9], cmap = 'gray')
    ax0 = fig.add_subplot(2,5,10)
    ax0.imshow(weights[0], cmap = 'gray')
        
    
    #plt.imshow(pixels, cmap='gray')
    plt.show()

    #what should happen
    #open weights
    #create a matrix scaled to 255 pixel values for each weight matrix
    #create matplotlib wwith 5x2 grid for all graphs
    #set each graph to a weight matrix
    #show all 10 weights across the 10 graphs


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



main()
displayData()
visualizeWeights()


