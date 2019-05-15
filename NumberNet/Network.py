import numpy as np
import time, pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#TODO break this file up

class Network():
    """
    Creates the network.
    Pickled after training to run data on it.
    """

    def __init__(self, parameters, layers):
        """
        Initializes the class with the given parameters and layers
        """
        self.parameters = parameters
        self.layers = layers

        ##MOVE THIS MEAN AND VARIANCE TO A SEPERATE METHOD
        #creates mean and variance when training to use
        self.meanImg = 255
        self.variance = 1

        #creates output lists, updated in trainBatch and pickled
        #after training is completed
        self.lossList = []
        self.batchAccuracyList = []
        self.accuracyList = []
       
        
    def forwardPass(self, image, labelIndex):
        """
        Forwards an image through the network.
        """
        #TODO REDO THIS

        self.vectors = [(np.array(image) - self.meanImg)/self.variance**.5]

        for layer in self.layers[:-1]:
            outputVector = layer[1].forwardPass(layer[0], self.vectors[-1])
            if layer[2] != None:
                outputVector = layer[2].forwardPass(outputVector)
            if len(layer) > 3:
                outputVector = layer[3].forwardPass(outputVector)
            self.vectors.append(outputVector)
        self.layers[-1].forwardPass(self.vectors[-1], labelIndex) #loss


    def backwardPass(self):
        """
        Backprops image through the network
        """
        #TODO REDO THIS
        #Layers store the values needed for computing gradients.
        #Gradient passes b/w layers must be hardcoded.
        #Retruns the gradients of the weights and biases (also hardcoded)
        grads = []

        localGrad = self.layers[-1].backwardPass() #loss grad, localGrad is priorGrad arg

        for i in range (2, len(self.layers)+1):
            layerGrad = []
            layer = self.layers[-i]
            if len(layer) > 3:
                localGrad = layer[3].backwardPass(localGrad)
            if layer[2] != None:
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


    def trainBatch(self, miniBatchImages, miniBatchLabels, regularization, decay, epoch):
        """
        trains the network on a single minibatch of data.
        Updates the gradient based on stepsize after computing
        the entire minibatch.
        """
        
        #creates list for weight gradients
        dW = []
        dB = []
        for layer in self.layers[:-1]:
            dW.append(np.zeros(layer[1].weights.shape))
            if (layer[2] != None):
                dB.append(np.zeros(layer[2].bias.shape))
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
            loss += self.layers[-1].loss

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
        self.parameters.stepSize = 1/(1 + decay * epoch) * self.parameters.initialStepSize
        for layer in self.layers[:-1]:
            layer[0].updateGrad(dW[layerNum], self.parameters)
            if layer[2] != None:
                layer[2].updateGrad(dB[layerNum], self.parameters)
            layerNum += 1

        #outputs data after training the minibatch

        #average loss and accuracy
        loss = loss / numData
        accuracy = accuracy / numData

        #add values to self lists to be pickled after training is complete.
        self.lossList.append(loss)
        self.batchAccuracyList.append(accuracy)

        #prints values to console so training can be seen
        #gives confidence the network isn't dead for the whole
        #train time.
        print('\nLoss: ', loss)
        print('Batch Accuracy: ' , accuracy)

   
    def train(self, trainImages, trainLabels, testImages, testLabels, parameters, batchSize, epochs):
        """
        Trains the network based on ALL train data
        and the parameters given.
        After this method is run the network is ready for use.
        """

        #records mean and variance of train data
        self.meanImg = np.mean(trainImages)
        self.variance = np.var(trainImages)

        #gets the paremeters to be used
        regularization = parameters.regularization
        decay = parameters.decay

        #times the network train time
        startTime = time.perf_counter()

        #Defines number of minibatches. If the minibatch isn't divisible by the
        #data size, it will round down and not run on all the train data.
        #Should be updated to be randomly ordered and have iterations
        numMinibatches = len(trainImages) // batchSize

        #opens loss files to output to
        lossFile = open('loss', 'wb')
        accuracyFile = open('accuracy', 'wb')

        #loops through data for specified number of times
        for x in range(epochs):
            
            #creates an index to use for slicing
            dataIndex = 0
            trainOrder = np.random.choice(len(trainImages), len(trainImages), replace = False) #returns a randomly ordered list of all indexes
            
            #trains for the number of minibatches in the whole dataset
            for i in range(numMinibatches):

                if(i % 10 == 0): #for every 100th batch, take a test accuracy sample
                    self.accuracyList.append((x * numMinibatches + i, self.accuracyTest(testImages, testLabels)))

                #miniBatch time tracker
                miniBatchStartTime = time.perf_counter()

                #slices train images and labels
                #miniBatchImages = trainImages[dataIndex : dataIndex+batchSize]
                #miniBatchLabels = trainLabels[dataIndex : dataIndex+batchSize]

                batchImages = []
                batchLabels = []
                batchIndex = trainOrder[dataIndex : dataIndex + batchSize]
                for index in batchIndex:
                    batchImages.append(trainImages[index])
                    batchLabels.append(trainLabels[index])

                #updates dataIndex
                dataIndex += batchSize

                #trains the minibatch in the trainBatch method.
                #data is added to the lists in this method.
                epoch = x
                self.trainBatch(batchImages, batchLabels, regularization, decay, epoch)

                #miniBatch time tracker
                miniBatchEndTime = time.perf_counter()

                #gets data on how weights are updating
                for layer in range(len(self.layers) - 1):
                    weight = self.layers[layer][0].weights
                    print('Variance of layer', layer + 1,':', np.var(weight))
                    
                #ouputs miniBatch time (sanity check while running)
                print('MiniBatch Time:', miniBatchEndTime - miniBatchStartTime)
                print('Epochs Remaining:', epochs - x - 1)
                print('Batches in Current Epoch Remaining:', numMinibatches - i)

        self.accuracyList.append((epochs* numMinibatches, self.accuracyTest(testImages, testLabels)))    

        #outputs data into files for analyzation
        pickle.dump(self.lossList, lossFile)
        pickle.dump(self.batchAccuracyList, accuracyFile)
        
        #closes files after data is output
        lossFile.close()
        accuracyFile.close()

        #times the networks total train time
        endTime = time.perf_counter()

        #outputs the train time to console
        print('Train time: ' , endTime - startTime)


    def displayData(self):
        """
        Displays accuracy and loss data.
        """

        #creates a matplotlib figure w/ 2 graphs for loss, accuracy
        fig, (lossPlot, accuracyPlot) = plt.subplots(1,2)

        #plots loss and accuracy
        lossPlot.plot(self.lossList)
        accuracyPlot.plot(self.batchAccuracyList, label = 'Batch Accuracy')
        testAccuracyX = [self.accuracyList[i][0] for i in range(len(self.accuracyList))]
        testAccuracyY = [self.accuracyList[i][1] for i in range(len(self.accuracyList))]
        accuracyPlot.plot(testAccuracyX, testAccuracyY, linestyle = 'dashed', label = 'Test Accuracy')
        accuracyPlot.legend()
        #shows plot
        plt.show()


    def visualizeWeights(self):
        """
        Vizualizes the weights in the network.
        Hard Coded so will only work with a known layer of weights.
        """

        #makes a list of np arrays for all weights
        weights = []
    
        #appends all 10 final weights to the list
        for i in range(10):
            weights.append(np.array(self.layers[0][0].weights[i]))
        
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


    def run(self, images, labels, delay = 2):
        """
        Runs the network to vizualize the data.
        Diplays a new image every n seconds
        Displays the image it is recognizing and the networks guess,
        as well as the correct answer.
        """
        #creates a matplotlib figure to show images
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

        #defines animate for matplotlib function
        def animate(i):

            #uses the 'i'th image of test data
            image = np.array(images[i])
            #reshapes image to 28x28 to be displayed
            image = np.reshape(image, (28,28))

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

        #creates the matplotlib animation, shows a new image every n seconds
        ani = animation.FuncAnimation(fig, animate, interval = delay*1000)

        plt.show()

    @property
    def scores(self):
        return self.layers[-1].scores