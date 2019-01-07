import Layers as ly
import numpy as np
from mnist import MNIST
import time
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Parameters():
    """
    Class used to pass hyperparameters for the network around
    using only one variable
    """

    def __init__(self, stepSize, regularization, miniBatchSize, epochs):
        self.stepSize = stepSize
        self.regularization = regularization
        self.miniBatchSize = miniBatchSize
        self.epochs = epochs

    #getters for the class
    def getStepSize(self):
        return self.stepSize

    def getRegularization(self):
        return self.regularization

    def getBatchSize(self):
        return self.miniBatchSize

    def getEpochs(self):
        return self.epochs


class Network():
    """
    Creates the network.
    Pickled after training to run data on it.
    """

    def __init__(self, parameters, layers):
        """
        Initializes the class with the given stepSize of the parameters
        The layers are also create here, and must be hardcoded into
        the network.
        """
        self.stepSize = parameters.getStepSize()
        
        self.layers = layers
        

    def forwardPass(self, image, labelIndex):
        """
        forwards an image through the network.
        Must be hardcoded in based on the network layers.
        """
        #layer = (Weights, Multiplication, Bias, Activation).....(loss)
        #self.vector = []
        self.vectors = [np.array(image) / 255]
        for layer in self.layers[:-1]:
            outputVector = layer[1].forwardPass(layer[0], self.vectors[-1])
            if len(layer) > 2:
                outputVector = layer[2].forwardPass(outputVector)
            if len(layer) > 3:
                outputVector = layer[3].forwardPass(outputVector)
            self.vectors.append(outputVector)
        self.layers[-1].forwardPass(self.vectors[-1], labelIndex) #loss



    def backwardPass(self):
        """
        performs a backwards pass through the network.
        Layers store the values needed for computing gradients.
        Gradient passes b/w layers must be hardcoded.
        Retruns the gradients of the weights and biases (also hardcoded)
        """
        grads = []

        localGrad = self.layers[-1].backwardPass() #loss grad, localGrad is priorGrad arg

        for i in range (2, len(self.layers)+1):
            layerGrad = []
            layer = self.layers[-i]
            if len(layer) > 3:
                localGrad = layer[3].backwardPass(localGrad)
            if len(layer) > 2:
                localGrad = layer[2].backwardPass(localGrad)
                layerGrad.insert(0, localGrad) #bias grad
            weightGrad, localGrad = layer[1].backwardPass(localGrad) #localGrad for next layer
            layerGrad.insert(0, weightGrad) #weightGrad
            grads.insert(0, layerGrad)

        return grads

    
    def accuracy(self, image, label):
        """
        tests the accuracy of a single image.
        This consists of doing a forward pass through the network
        and returning a 0 or 1 for accuracy (true/false)
        """
        
        self.forwardPass(image, label)
        scores = self.vectors[-1]
        largestValue = 0 #index of largest value starts at 0
       
        for j in range(1, len(scores)):
            #finds the index of the largest score for all scores
            if (scores[j] > scores[largestValue]):
                #if there is a new larger value, updates index
                largestValue = j
                
        if (largestValue == label):
            #when the highest value score is correct
            return 1
        else:
            return 0
        
    
    def accuracyTest(self, testImages, testLabels):
        """
        defines the accuracy of the network based on test Data.
        Return the accuracy as a decimal out of 1.0 (1.0 = 100%)
        """
        
        #accuracy starts at 0, adds 1 for each correctly identified image
        accuracy = 0.0

        #loops through all test data adding to accuracy
        for i in range(len(testImages)):
            accuracy += self.accuracy(testImages[i], testLabels[i])           
        
        #compute % accuracy
        return accuracy / len(testImages)


    def trainBatch(self, miniBatchImages, miniBatchLabels, regularization):
        """
        trains the network on a single minibatch of data.
        Updates the gradient based on stepsize after computing
        the entire minibatch.
        """
        
        #creates list for weight gradients
        dW = []
        dB = []
        for layer in self.layers[:-1]:
            dW.append(np.zeros(layer[1].getWeights().shape))
            if len(layer) > 2:
                if (layer[2].getBias().any() != None):
                    dB.append(np.zeros(layer[2].getBias().shape))
                else:
                    dB.append(None)
        #stores number of images being tested
        numData = len(miniBatchImages)

        #tracks accuracy, starts at 0
        accuracy = 0.0
        #tracks loss, starting at 0
        loss = 0.0

        #runs through the miniBatch
        for i in range(numData):
            
            #forwards a single image and label through the network
            #inside the accuracy method.
            #Adds to accuracy after forwards pass is complete
            accuracy += self.accuracy(miniBatchImages[i], miniBatchLabels[i])

            #updates loss
            loss += self.layers[-1].getLoss()

            #backprops and adds to weights and biases
            grads = self.backwardPass()
            for i in range(len(self.layers) - 1):
                dW[i] += grads[i][0]
                if(dB[i].any() != None):
                    dB[i] += grads[i][1]

        #avg all gradients of the minibatch
        for weight in dW:
            weight /= numData
        for bias in dB:
            if bias.any() != None:
                bias /= numData

        #update weights for minibatch gradients
        layerNum = 0
        for layer in self.layers[:-1]:
            layer[0].updateGrad(self.stepSize, dW[layerNum], regularization)
            if layer[2].getBias().any() != None:
                layer[2].updateGrad(self.stepSize, dB[layerNum])
            layerNum += 1

        #outputs data after training the minibatch

        #average loss and accuracy
        loss = loss / numData
        accuracy = accuracy / numData

        #add values to self lists to be pickled after training is complete.
        self.lossList.append(loss)
        self.accuracyList.append(accuracy)
        self.layersList.append(self.layers)

        #prints values to console so training can be seen
        #gives confidence the network isn't dead for the whole
        #train time.
        print('\nLoss: ', loss)
        print('Accuracy: ' , accuracy)


   
    def train(self, trainImages, trainLabels, parameters):
        """
        Trains the network based on ALL train data
        and the parameters given.
        After this method is run the network is ready for use.
        """
        #gets the paremeters to be used
        batchSize = parameters.getBatchSize()
        epochs = parameters.getEpochs()
        regularization = parameters.getRegularization()

        #times the network train time
        startTime = time.perf_counter()

        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        #Should be updated to be randomly ordered and have iterations
        numMinibatches = len(trainImages) // batchSize

        #creates output lists, updated in trainBatch and pickled
        #after training is completed
        self.lossList = []
        self.accuracyList = []
        self.layersList = []

        #opens loss files to output to
        lossFile = open('loss', 'wb')
        accuracyFile = open('accuracy', 'wb')
        layersFile = open('layers', 'wb')

        #adds initialized layers to file (so layers before training can be observed)
        self.layersList.append(self.layers)
            
        #loops through data for specified number of times
        for x in range(epochs):
            
            #creates an index to use for slicing
            dataIndex = 0
            
            #trains for the number of minibatches in the whole dataset
            for i in range(numMinibatches):

                #miniBatch time tracker
                miniBatchStartTime = time.perf_counter()

                #slices train images and labels
                miniBatchImages = trainImages[dataIndex : dataIndex+batchSize]
                miniBatchLabels = trainLabels[dataIndex : dataIndex+batchSize]

                #updates dataIndex
                dataIndex += batchSize

                #trains the minibatch in the trainBatch method.
                #data is added to the lists in this method.
                self.trainBatch(miniBatchImages, miniBatchLabels, regularization)

                #miniBatch time tracker
                miniBatchEndTime = time.perf_counter()
                #ouputs miniBatch time (sanity check while running)
                print('MiniBatch Time:', miniBatchEndTime - miniBatchStartTime)
                print('Epochs Remaining:', epochs - x - 1)
                print('Batches in Current Epoch Remaining:', numMinibatches - i)


        #outputs data into files for analyzation
        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.accuracyList, accuracyFile)
        pickle.dump(self.layersList, layersFile)
        
        #closes files after data is output
        lossFile.close()
        accuracyFile.close()
        layersFile.close()

        #times the networks total train time
        endTime = time.perf_counter()

        #outputs the train time to console
        print('Train time: ' , endTime - startTime)

    def getScores(self):
        return self.layers[-1].getScores()



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


def displayData():
    """
    Displays the data (besided weights) that wasn't logged during training.
    Shows accuracy and loss.
    """

    #opens pickled log files created during training
    try:
        accuracy = open('accuracy', 'rb')
        loss = open('loss', 'rb')
    except FileNotFoundError:
        print('Log files don\'t exist')
        return None

    #recreates the pickled objects
    try:
        accuracyList = pickle.load(accuracy)
        lossList = pickle.load(loss)
    except:
        print('Invalid log files.')
        return None

    #closes pickle files once objects have been created.
    accuracy.close()
    loss.close()

    #creates a matplotlib figure w/ 2 graphs for loss, accuracy
    fig, (lossPlot, accuracyPlot) = plt.subplots(1,2)

    #plots loss and accuracy
    lossPlot.plot(lossList)
    accuracyPlot.plot(accuracyList)

    #shows plot
    plt.show()


def trainNetwork(parameters, layers):
    """
    Trains the network and pickles it after it is trained.
    Once trained, the network can be run on data without needing to be trained again.
    """

    #init network
    numberNet = Network(parameters, layers)

    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #test initial accuracy on test data
    initialAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #trains the network
    numberNet.train(trainImages, trainLabels, parameters)

    #tests final accuracy on test data
    finalAccuracy = numberNet.accuracyTest(testImages, testLabels)

    #display output data
    print('Bias: ', numberNet.layers[0][2].getBias())
    print('initAccuracy: ', initialAccuracy)
    print('finalAccuracy: ', finalAccuracy)

    #pickles network for re-use
    networkFile = open('network', 'wb')
    pickle.dump(numberNet, networkFile)
    networkFile.close()


def visualizeWeights():
    """
    Vizualizes the weights in the network.
    Hard Coded so will only work with a known layer of weights.
    """

    #load weights as weightsList from pickled file
    try:
        weights = open('weights', 'rb')
        weightsList = pickle.load(weights)
        weights.close()
    except:
        print('Invalid or missing file')
        return None

    #makes a list of np arrays for all weights
    weights = []
    
    #appends all 10 final weights to the list
    for i in range(10):
        weights.append(np.array (weightsList[len(weightsList)-1][i]) )
        
    #finds max value to scale images
    #uses absolute values to avoid negatives
    maxPixel = np.amax(np.abs(weights))

    #scales array to 255.0 pixel values
    scalar = 255.0 / maxPixel
    
    for i in range(len(weights)):
        #scales weights to 255.0 values
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


def runNetwork(n):
    """
    Runs the network to vizualize the data.
    Diplays a new image every n seconds
    Displays the image it is recognizing and the networks guess,
    as well as the correct answer.
    """
    
    #import data
    trainImages, trainLabels, testImages, testLabels = importData()

    #opens network as 'network'
    try:
        networkFile = open('network', 'rb')
        network = pickle.load(networkFile)
        networkFile.close()
    except:
        print('Cannot open or load network file.')
        return None

    #creates a matplotlib figure to show images
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    #defines animate for matplotlib function
    def animate(i):

        #uses the 'i'th image of test data
        image = np.array(testImages[i])
        #reshapes image to 28x28 to be displayed
        image = np.reshape(image, (28,28))

        #runs a forwardPass, the trainLabel is needed for computing the loss (in forward pass method)
        #if the network is intened to be run w/o known labels, could put any number in for the test label
        #and it would still give an unaffected guess.
        network.forwardPass(testImages[i], testLabels[i])

        #gets scores from network and picks the max index as the number guess
        scores = np.array(network.getScores())
        maxScoreIndex = np.argmax(scores)

        #creates a title for the image showing the guess and true value
        title = 'Network Guess: ' + str(maxScoreIndex) + ' Actual: ' + str(testLabels[i])

        #plots the image
        ax1.clear()
        ax1.set_title(title)
        ax1.imshow(image, cmap = 'gray')

    #creates the matplotlib animation, shows a new image every n seconds
    ani = animation.FuncAnimation(fig, animate, interval = n*1000)

    plt.show()
        
def createLayer(input, output, biasSize = False, activationFunction = ""):
    layerList = []
    layerList.append(ly.Weights(output, input))
    layerList.append(ly.Multiplication())
    if (biasSize == False):
        layerList.append(ly.NoBias())
    else:
        layerList.append(ly.Bias(output))
    if(activationFunction != ""):
        if activationFunction == 'ReLU':
            layerList.append(ly.ReLU())

    return layerList
    

#layer = (Weights, Multiplication, Bias, Activation).....(loss)
layers = [createLayer(784, 10, True, 'ReLU'), ly.Softmax()]
#layers = [ (ly.Weights(10,784), ly.Multiplication(), ly.Bias(10)), ly.Softmax()]
#Trains Network
parameters = Parameters(stepSize = 1e-3, regularization = .4, miniBatchSize= 2500, epochs = 1)
trainNetwork(parameters, layers)

#displays the data on the network training
displayData()

#vizualizes the netowrks weights
visualizeWeights()

#runs the network visually
runNetwork(2)
