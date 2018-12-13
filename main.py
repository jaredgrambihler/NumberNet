import Layers as ly
import numpy as np
from mnist import MNIST
import time
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation



#class to create the network and store it in memory. Can be pickled to avoid retraining the network.
class Network():

    #initializes the class with an optionally defined step size.
    #the layers that are stored must be written here, although
    #the interactions b/w layers are in the forward and backward pass
    #methods.
    def __init__(self, stepSize = 10e-7):
        
        #init weights, size is (labelNumbers, imagePixels)
        self.w1 = ly.Weights(10, 784)
        
        #init Biases
        #could be added onto weights matrix later on
        self.b1 = ly.Bias(10)
        
        #init network layers
        #this is a single layer network w/ no activation function.
        self.l1 = ly.Multiplication()
        self.loss = ly.Softmax()
        self.stepSize = stepSize
        

    #forwards an image throught the network.
    #saves the score outputs and loss to the class.
    #could be made to work with a list as the layers of the network
    def forwardPass(self, image, labelIndex):
        
        #layer1 forward pass, happens to be only layer.
        self.h1 = self.l1.forwardPass(self.w1, image)
        self.h2 = self.b1.forwardPass(self.h1)

        #loss function forward pass.
        self.loss.forwardPass(self.h2, labelIndex)


    #backprops through the network, using values stored in the layers.
    #returns the gradient of the weight classes.
    #could be made to work with a list as layers of the network.
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
        
    
    #defines the accuracy of the network on the test data (10k images/labels)
    #returns the accuracy as a decimal value (1.0 is 100%)
    def accuracyTest(self, testImages, testLabels):
        
        #accuracy starts at 0, adds 1 for each correctly identified image
        accuracy = 0.0

        ##loops through all test data
        for i in range(len(testImages)):
            accuracy += self.accuracy(testImages[i], testLabels[i])           
        
        #compute % accuracy
        return accuracy / len(testImages)


    #trains the network on a minibatch of data.
    #updates the gradient after find the avg gradient for the
    #entire minibatch.
    def trainBatch(self, miniBatchImages, miniBatchLabels):
        
        #initializes return array the same size as weights and biases
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
            #adds to accuracy after forwards pass is complete
            accuracy += self.accuracy(miniBatchImages[i], miniBatchLabels[i])

            #updates loss
            loss += self.loss.loss

            #backprops and adds to weights and biases
            biasGrad, weightGrad = self.backwardPass()
            dW += weightGrad
            dB += biasGrad

        #avg all gradients of the minibatch
        dW = dW / numData
        dB = dB / numData

        #update weights for minibatch gradients
        self.w1.updateGrad(self.stepSize, dW)
        self.b1.updateGrad(self.stepSize, dB)

        #outputs data after training the minibatch
        #potential to make this a seperate method for output data

        #average loss and accuracy
        loss = loss / numData
        accuracy = accuracy / numData

        #add values to self lists to be pickled after training is complete.
        self.lossList.append(loss)
        self.accuracyList.append(accuracy)
        self.weightsList.append(self.w1.weights)

        #prints values to console so training can be seen
        #gives confidence the network isn't dead for the whole
        #train time.
        print('Loss: ', loss)
        print('Accuracy: ' , accuracy)
        print('Weights: ' , self.w1.weights)


    #trains the network. Takes in train data and optional batch size
    #for a specified number of epochs and outputs data on the network
    #for each minibatch.
    def train(self, trainImages, trainLabels, batchSize = 2500, epochs = 2):

        #times the network train time
        startTime = time.perf_counter()

        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        #Should be updated to be randomly ordered and have iterations
        numMinibatches = int(len(trainImages)/batchSize)

        #creates output lists, updated in trainBatch and pickled
        #after training is completed
        self.lossList = []
        self.accuracyList = []
        self.weightsList = []

        #opens loss files to output to
        lossFile = open('loss', 'wb')
        accuracyFile = open('accuracy', 'wb')
        weightsFile = open('weights', 'wb')

        #adds initialized weights to file (so weights before training can be observed)
        self.weightsList.append(self.w1.weights)
            
        #loops through data for specified number of times
        for x in range(epochs):
            
            #creates an index to use for slicing
            dataIndex = 0
            
            #trains batches
            for i in range(int(len(trainImages)/batchSize)):

                #miniBatch time tracker
                miniBatchStartTime = time.perf_counter()

                #slices train images and labels
                miniBatchImages = trainImages[dataIndex : dataIndex+batchSize]
                miniBatchLabels = trainLabels[dataIndex : dataIndex+batchSize]

                #updates dataIndex
                dataIndex += batchSize

                #trains the minibatch
                self.trainBatch(miniBatchImages, miniBatchLabels)

                #miniBatch time tracker
                miniBatchEndTime = time.perf_counter()
                #ouputs miniBatch time (sanity check while running)
                print(miniBatchEndTime - miniBatchStartTime)


        #outputs data into files
        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.accuracyList, accuracyFile)
        pickle.dump(self.weightsList, weightsFile)
        
        #closes files after data is output
        lossFile.close()
        accuracyFile.close()
        weightsFile.close()

        #times the networks total train time
        endTime = time.perf_counter()

        #outputs the train time to console
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
#needs work
def displayData():

    #opens pickled log files created during training
    accuracy = open('accuracy', 'rb')
    loss = open('loss', 'rb')
    weights = open('weights', 'rb')

    #recreates the pickled objects
    accuracyList = pickle.load(accuracy)
    lossList = pickle.load(loss)
    weightsList = pickle.load(weights)

    #closes pickle files once objects have been created.
    accuracy.close()
    loss.close()
    weights.close()

    #creates a matplotlib figure w/ 4 graphs for
    #loss, accuracy, intialized weights, and final weights
    fig, (lossPlot, accuracyPlot, weightHist0, weightHist1) = plt.subplots(1,4)

    #plots loss and accuracy
    lossPlot.plot(lossList)
    accuracyPlot.plot(accuracyList)

    #reshapes first and last weights to be a list
    weights0 = weightsList[0].reshape(7840)
    weights1 = weightsList[len(weightsList) -1].reshape(7840)

    #plots weights as histograms
    weightHist0.hist(weights0, bins = 20)
    weightHist1.hist(weights1, bins = 20)

    #shows plot
    plt.show()


#this should become a method for training the network, which pickles the network
#when it is completed so the network can be used again w/o training
def trainNetwork():

    #init network
    numberNet = Network()

    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #test initial accuracy
    initialAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #trains the network
    numberNet.train(trainImages, trainLabels)

    #tests final accuracy
    finalAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #display output data
    print('Bias: ', numberNet.b1.bias)
    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)

    #pickles network for re-use
    networkFile = open('network', 'wb')
    pickle.dump(numberNet, networkFile)
    networkFile.close()


#vizualizes the weights in a network
def visualizeWeights():

    #load weights as weightsList from pickled file
    weights = open('weights', 'rb')
    weightsList = pickle.load(weights)
    weights.close()

    #makes a list of np arrays for all weights
    weights = []
    
    #appends all 10 final weights to the list
    for i in range(10):
        weights.append(np.array (weightsList[len(weightsList)-1][i]) )
        
    #finds max to scale images
    #could use numpy here to be more efficient
    maxPixel = 0
    
    #loops through all elements of the weights
    for weight in weights:
        for element in weight:
            
            #finds max value in ALL weights (shows importance among all scores equally)
            if (abs(element) > maxPixel):
                maxPixel = abs(element)

    #scales array to 255.0 pixel values
    scalar = 255.0 / maxPixel
    
    for i in range(len(weights)):
        weights[i] *= scalar
        
        #reshapes the weights to be a 28x28 image
        weights[i] = np.reshape(weights[i], (28,28))

    #creates a plot for all weights
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
        
    #shows weights
    plt.show()


#runs network and displays labels and images
def runNetwork():
    
    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #opens network as 'network'
    networkFile = open('network', 'rb')
    network = pickle.load(networkFile)
    networkFile.close()

    #creates a matplotlib figure to show images
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)

    #defines animate for matplotlib function
    def animate(i):

        #uses the image of i times 
        image = np.array(testImages[i])
        image = np.reshape(image, (28,28))

        #runs a forwardPass, the trainLabel is needed so the function
        #works although this could be changed
        network.forwardPass(testImages[i], testLabels[i])

        #gets scores from network and picks the max index as the number guess
        scores = np.array(network.loss.probScores)
        maxScoreIndex = np.argmax(scores)

        #creates a title for the image showing the guess and true value
        title = 'Network Guess: ' + str(maxScoreIndex) + ' Actual: ' + str(testLabels[i])

        #plots the image
        ax1.clear()
        ax1.set_title(title)
        ax1.imshow(image, cmap = 'gray')

    #creates the matplotlib animation, shows a new image every 2 seconds
    ani = animation.FuncAnimation(fig, animate, interval = 2000)

    plt.show()
        

#this could be done better, just some code to run the network
#and display the results
#trainNetwork()
displayData()
visualizeWeights()
runNetwork()
