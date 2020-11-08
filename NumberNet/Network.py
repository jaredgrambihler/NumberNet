"""
Contains Network class.
"""
import numpy as np
import time, pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Network():
    """
    Class that manages the entire network.
    """

    def __init__(self, parameters, layers, lossFunction):
        """
        Initialize network with given arguments.
        Args:
            parameters (Paramaters object): used as parameters during network training
            layers (List): list of Layer objects that define the network
            lossFunction (ActivationFunction class): loss function to be used
        """
        self.parameters = parameters
        self.layers = layers
        self.lossFunction = lossFunction 

        #creates output lists, updated in trainBatch and pickled after training is completed
        self.lossList = []
        self.batchAccuracyList = []
        self.accuracyList = []
       
    def setNormalizeData(self, trainImages):
        """
        Sets means and variance to be used for normalizing data
        Args:
            trainImages (numpy array): List of 1D numpy arrays used for training network.
        """
        trainImages = np.array(trainImages)
        self.meanImg = np.zeros(len(trainImages[0]))
        for i in range(len(trainImages[0])):
            self.meanImg[i] = np.mean(trainImages[:,i])
        self.variance = np.var(trainImages)

    def forwardPass(self, data, labelIndex):
        """
        Forwards an image through the network.
        Args:
            data (numpy array): 1D array to be forwarded
            labelIndex (int): truthy label for the given data
        Returns:
            numpy array: Scores vector output by the final layer of the network.
        """
        currentVector = data
        for layer in self.layers:
            currentVector = layer.forwardPass(currentVector)
        currentVector = self.lossFunction.forwardPass(currentVector, labelIndex) #loss
        return currentVector #scores


    def backwardPass(self):
        """
        Backprops image through the network
        Updates weights of network based on backpropagation.
        """
        #Layers store the values needed for computing gradients.
        #Weights and bias gradients stored in each layer
        #Updating of weights is done inside each layer.
        grads = []
        currentGrad = self.lossFunction.backwardPass() #loss grad, localGrad is priorGrad arg
        for i in range(len(self.layers)):
            currentGrad = self.layers[-(i+1)].backwardPass(currentGrad)
    

    def accuracy(self, data, label):
        """
        Tests the accuracy of a single image by running a forward pass through the network.
        Args:
            data (numpy array): 1D array to evaluate accuracy on
            label (int): truthy label of data
        Returns:
            int: 1 for correctly indetifying label, 0 for being wrong
        """
        scores = self.forwardPass(data, label)
        largestValue = 0 #index of largest value starts at 0
        #finds the index of the largest score for all scores
        for j in range(1, len(scores)): #checks against all scores
            if (scores[j] > scores[largestValue]): #if there is a new larger value, updates index
                largestValue = j
        if (largestValue == label): #when the highest value score is correct
            return 1
        else:
            return 0

   
    def accuracyTest(self, testImages, testLabels):
        """
        Defines the accuracy of the network based on the test data.
        Args:
            testImages (numpyArray): 2D numpy array. Each index should contain a numpy array that is a 
                single input vector.
            testLabels (list): list containing int labels. testLabels[i] label should correspond to the data
                at testImages[i]
        Returns:
            float: 0-1 float stating the accuracy of the network evaluated on the test data.
        """
        accuracy = 0 #accuracy starts at 0, adds 1 for each correctly identified image
        #loops through all test data adding to accuracy
        for i in range(len(testImages)):
            accuracy += self.accuracy(testImages[i], testLabels[i])           
        return float(accuracy) / len(testImages) #compute % accuracy


    def trainBatch(self, miniBatchImages, miniBatchLabels, epoch):
        """
        Trains the network on a single minibatch of data.
        Updates the gradient based on stepsize after computing
        the entire minibatch.
        Args:
            miniBatchImages (numpy array): 2D array where each index contains a 1D numpy array which is a single
                input vector.
            miniBatchLabels (list): list of int labels. miniBatchLabels[i] should correspond to the data at
                miniBatchImages[i]
            epoch (int): current epoch that is being run
        """
        numData = len(miniBatchImages)  #stores number of images being tested
        accuracy = 0.0                  #tracks accuracy, starts at 0
        loss = 0.0                      #tracks loss, starting at 0
        #runs through the miniBatch
        for i in range(numData):
            #Forwards a single image and label through the network inside the accuracy method.
            #Adds to accuracy after forwards pass is complete
            adjImage = np.array(miniBatchImages[i]) - self.meanImg
            accuracy += self.accuracy(adjImage, miniBatchLabels[i])
            loss += self.lossFunction.loss #updates loss
            self.backwardPass() #backprops and adds to weights and biases
        #update weights for minibatch gradients
        self.parameters.stepSize = 1/(1 + self.parameters.decay * epoch) * self.parameters.initialStepSize
        for layer in self.layers:
            layer.updateGrad(numData, self.parameters)
        #average loss and accuracy
        loss = loss / numData
        accuracy = accuracy / numData
        #add values to self lists to be stored
        self.lossList.append(loss)
        self.batchAccuracyList.append(accuracy)
        #prints values to console so training can be seen
        #gives confidence the network is working throughout training
        print('\nBatch Loss: ', loss)
        print('Batch Accuracy: ' , accuracy)

   
    def train(self, trainImages, trainLabels, testImages, testLabels, batchSize, epochs):
        """
        Trains the network based on all train data and the parameters given.
        Prints data regarding networks status throughout training. Saves loss and accuracy data after training
        to pickle files. After this method is run the network is ready for use.
        Args:
            trainImages (numpy array): List where each element is a 1D numpy array. Each numpy array must be of equal
                length and of the same dimension as the first layer of the network.
            trainLabels (list): List where each element is a label (int) corresponding to the train images. Indexes
                must correspond to trainImages (e.g. trainLabels[i] is the label for trainImages[i])
            testImages (numpy array): List where each element is a 1D numpy array. Used for testing the accuracy of the
                network. Numpy arrays must be of the same dimensions as those in trainImages.
            testLabels (list): Similar to trainLabels, but for the testImages list.
            batchSize (int): Size to be used for minibatches when training
            epochs (int): Number of epochs to run the network when training
        """
        self.setNormalizeData(trainImages) #records mean and variance of train data
        startTime = time.perf_counter() #used to time training
        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        numMinibatches = len(trainImages) // batchSize
        for x in range(epochs):
            dataIndex = 0  #index to use for slicing
            trainOrder = np.random.choice(len(trainImages),
                                          len(trainImages),
                                          replace = False) #returns a randomly ordered list of all indexes
            for i in range(numMinibatches): #trains for the number of minibatches in the whole dataset
                if(i % 100 == 0): #for every 100th batch, take a test accuracy sample
                    self.accuracyList.append((x * numMinibatches + i, self.accuracyTest(testImages, testLabels)))
                miniBatchStartTime = time.perf_counter() #miniBatch time tracker
                #slices train images and labels, image indexes for the current batch come from trainOrder list
                batchImages = []
                batchLabels = []
                batchIndex = trainOrder[dataIndex : dataIndex + batchSize] 
                for index in batchIndex: #adds images for minibatch to the batch lists
                    batchImages.append(trainImages[index])
                    batchLabels.append(trainLabels[index])
                dataIndex += batchSize #updates dataIndex for next batch
                #trains the minibatch in the trainBatch method. Data is added to the lists in this method.
                self.trainBatch(batchImages, batchLabels, x) # x is current epoch
                miniBatchEndTime = time.perf_counter() #miniBatch time tracker
                #gets data on how weights are updating
                layerNum = 1
                for layer in self.layers[:-1]:
                    print('Variance of layer', layerNum,':', np.var(layer.weights.weights))
                    layerNum += 1
                #ouputs miniBatch time (sanity check while running)
                print('MiniBatch Time:', miniBatchEndTime - miniBatchStartTime)
                print('Epochs Remaining:', epochs - x - 1)
                print('Batches in Current Epoch Remaining:', numMinibatches - i)        
        #opens loss files to output to
        lossFile = open('loss', 'wb')
        accuracyFile = open('accuracy', 'wb')
        #outputs data into files for analyzation
        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.batchAccuracyList, accuracyFile)
        #closes files after data is output
        lossFile.close()
        accuracyFile.close()
        endTime = time.perf_counter() #times the networks total train time
        print('Total Train time: ' , endTime - startTime) #outputs the train time to console


    def displayData(self):
        """
        Displays accuracy and loss data in matplotlib figures
        """
        #creates a matplotlib figure w/ 2 graphs
        #one is for loss, one is for accuracy
        fig, (lossPlot, accuracyPlot) = plt.subplots(1,2)
        #plots loss and accuracy
        lossPlot.plot(self.lossList)
        accuracyPlot.plot(self.batchAccuracyList, label = 'Batch Accuracy')
        testAccuracyX = [self.accuracyList[i][0] for i in range(len(self.accuracyList))] #get minibatch number
        testAccuracyY = [self.accuracyList[i][1] for i in range(len(self.accuracyList))] #get accuracy value
        accuracyPlot.plot(testAccuracyX, testAccuracyY, linestyle = 'dashed', label = 'Test Accuracy')
        accuracyPlot.legend()
        plt.show()


    def visualizeWeights(self):
        """
        Vizualizes the weights in a single layer network using MNIST data.
        Hard coded for intended use, will likely be meaningless on other data/networks
        """
        weights = [] #for list of np arrays for all weights
        #appends all 10 final weights to the list
        for i in range(len(self.layers[0].weights.weights)):
            weights.append(self.layers[0].weights.weights[i])
        #finds max value to scale images
        #uses absolute values to avoid negatives
        maxPixel = np.amax(np.abs(weights))
        scalar = 255.0 / maxPixel #scales array to 255.0 pixel values
        for i in range(len(weights)):
            weights[i] *= scalar #scales weights to 255.0 values
            weights[i] = np.reshape(weights[i], (28,28)) #reshapes the weights to be a 28x28 image
        fig = plt.figure() #creates a plot for all weights
        #puts each weight visualization on a subplot
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
        plt.show()


    def run(self, images, labels, delay = 2):
        """
        Runs the network to vizualize the data. Displays network guess and true label.
        Hard coded to work with MNIST digits
        Args:
            images (list): list that contains each image to be evaluated. Images can be lists or numpy arrays
            labels: list that contains labels corresponding to images. labels[i] should correspond to images[i]
            delay (int): Default value of 2. Number of seconds between each forward pass that is performed.
        """
        #creates a matplotlib figure to show images
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        #defines animate for matplotlib function
        def animate(i):
            image = np.array(images[i]) #uses the 'i'th image of test data
            image = np.reshape(image, (28,28)) #reshapes image to 28x28 to be displayed
            #runs a forwardPass, the trainLabel is needed for computing the loss (in forward pass method)
            #if the network is intened to be run w/o known labels, could put any number in for the test label
            #and it would still give an unaffected guess.
            self.forwardPass(images[i], labels[i])
            #gets scores from network and picks the max index as the number guess
            scores = np.array(self.scores)
            maxScoreIndex = np.argmax(scores)
            #creates a title for the image showing the guess and true value
            title = 'Network Guess: ' + str(maxScoreIndex) + ' Actual: ' + str(labels[i])
            #plots the image
            ax1.clear()
            ax1.set_title(title)
            ax1.imshow(image, cmap = 'gray')
        #creates the matplotlib animation, shows a new image every "delay" seconds
        ani = animation.FuncAnimation(fig, animate, interval = delay*1000)
        plt.show()

    @property
    def scores(self):
        """
        Probability scores determined by loss function on most recently forwarded image.
        """
        return self.lossFunction.probScores