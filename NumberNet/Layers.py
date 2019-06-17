import numpy as np
import math
from . import ActivationFunctions

class Layer:
    """
    Class to be defined for each layer of the network.
    """
    #---Layer format---
    #Weights
    #Biases (can move to weights later to optimize)
    #Activation Function (can be None)

    def __init__(self, inputSize, outputSize, bias = False, activationFunction = None):
        #TODO - for bias and activation function instead of setting to None create versions of those classes w/ no impact to simplify code.
        self.weights = Weights(inputSize, outputSize)
        if(bias):
            self.bias = Bias(outputSize) #DETERMINE IF BIAS IS FORWARD/BACKWARD HERE OR IN SEPERATE CLASS
        else:
            self.bias = None
        #Sets activation function
        if(activationFunction):
            if activationFunction == 'ReLU':
                self.activationFunction = ActivationFunctions.ReLU()
            elif activationFunction == 'Sigmoid':
                self.activationFunction = ActivationFunctions.Sigmoid()
            elif activationFunction == 'LeakyReLU' or activationFunction == 'HipsterReLU':
                self.activationFunction = ActivationFunctions.LeakyReLU()
            else:
                raise Exception #invalid activation function
        else:
            self.activationFunction = None

    #FORWARD AND BACKWARD PASS FROM MULTIPLICATION LAYER
    #should handle the forward or backward pass of each layer, taking in the input and returning the output for forward. Takes in gradient and returns it for backwards.
    def forwardPass(self, inputVector):
        """
        stores weights and input for the backward pass.
        Returns a dot product between them
        """
        weightOutput = self.weights.forwardPass(inputVector)
        if(self.bias):
            biasOutput = weightOutput + self.bias.bias
        else:
            biasOuput = weightOutput
        if(self.activationFunction):
            return self.activationFunction.forwardPass(biasOutput)
        else:
            return biasOutput

    def backwardPass(self, priorGradient):
        """
        Computes gradients of both the weights and the input
        Returns weightGrad, inputGrad
        """
        if(self.activationFunction):
            priorGradient = self.activationFunction.backwardPass(priorGradient)
        if(self.bias):
            priorGradient = self.bias.backwardPass(priorGradient)
        priorGradient = self.weights.backwardPass(priorGradient)
        return priorGradient

    def updateGrad(self, numData, parameters):
        #method to upgrade gradients of weights in biases in the layer
        self.weights.updateGrad(numData, parameters)
        self.bias.updateGrad(numData, parameters)


class Weights:
    """Class for weights on each layer"""
    #ADD STORAGE OF GRADIENTS FOR EACH FORWARD PASS TO BE USED IN UPDATEGRAD
    def __init__(self, inputSize, outputSize):
        """
        Creates weights based on the input size.
        x = number of outputs
        y = number of inputs
        """
        #guassian distribution w/ sd of sqrt(2/inputs)
        self._weights = np.random.randn(inputSize*outputSize) * math.sqrt(2.0/inputSize)
        self._weights = np.reshape(self._weights, (outputSize, inputSize))

        #learning params
        self.vdw = np.zeros((outputSize, inputSize))
        self.sdw = np.zeros((outputSize, inputSize))
        self.grad = np.zeros((outputSize, inputSize))


    def forwardPass(self, inputVector):
        self.input = inputVector
        return np.dot(self._weights, inputVector)


    def backwardPass(self, priorGradient):
        #creates matrices of the proper size to hold the gradients
        weightGrad = np.ones(np.shape(self._weights))
        inputGrad = np.ones(np.shape(self.input))

        for i in range(len(self.input)):
            #mutiplies the input at i to the entire row
            #of weights it ends up impacting
            tempGrad = self.input[i]
            weightGrad[: , i] = tempGrad
        for i in range(len(priorGradient)):
            #takes the priorGradient at i and multiplies
            #it to the entire row that was a part of it's result
            weightGrad[i] *= priorGradient[i]

        #creates the inputGrad
        for i in range(len(self.input)):
            #takes the sum of the weightsRow that impacted the inputGrad's effect
            weightRow = self._weights[:, i]
            #multiplies together the row each priorGradient that it was related t0
            tempGrad = np.multiply(weightRow, priorGradient)
            #takes the sum of the 'i' row impact as the local gradient 
            #(already multiplied to the prior gradient)
            inputGrad[i] = np.sum(tempGrad)

        self.grad += weightGrad
        return inputGrad


    def updateGrad(self, numData, parameters):
        """
        Updates the weights based on gradient, stepSize, and Regularization.
        Should be called after the avg gradient is computed for a minibatch
        """
        #momentum and RMSProp implemented w/o correction
        self.grad /= numData
        if parameters.momentum == True:
            self.vdw = parameters.beta1 * self.vdw + (1- parameters.beta1) * self.grad
            self._weights -= parameters.stepSize * self.vdw
        else:
            self._weights -= parameters.stepSize * self.grad

        if parameters.RMSProp == True:
            self.sdw = parameters.beta2 * self.sdw + (1 - parameters.beta2) * self.grad**2
            self._weights /= (self.sdw + parameters.epsilon)**.5

        #regularization function of L2 Regularization
        #Reference : http://cs231n.github.io/neural-networks-2/
        self._weights -= parameters.regularization * self.weights
        self.grad *= 0 #reset for next batch

    @property
    def weights(self):
        """
        Returns the weights
        """
        return self._weights


class Bias:
    """
    Layer for biases.
    Should be implemented after weights and input are multiplied together.
    """
    #ADD STORAGE OF BIAS GRAD

    def __init__(self, x):
        """
        Creates biases of length 'n' to all be zero
        """
        self._bias = np.ones(x)
        self.vdb = np.zeros(x)
        self.sdb = np.zeros(x)
        self.grad = np.zeros(x)


    def backwardPass(self, priorGradient):
        """
        No change to gradient in addition. Saves gradient for bias update.
        """
        self.grad += priorGradient
        return priorGradient


    def updateGrad(self, numData, parameters):
        """
        Performs an update to the biases.
        Should be done after the avg is computes from a minibatch
        """
        self.grad /= numData
        if parameters.momentum == True:
            self.vdb = parameters.beta1 * self.vdb + (1 - parameters.beta1) * self.grad
            self._bias -= self.vdb * parameters.stepSize
        else:
            self._bias -= self.grad * parameters.stepSize

        if parameters.RMSProp == True:
            self.sdb = parameters.beta2 * self.sdb + (1 - parameters.beta2) * self.grad**2
            self._bias /= (self.sdb + parameters.epsilon)**.5
        self.grad *= 0 #reset grad


    @property
    def bias(self):
        """
        Returns the biases
        """
        return self._bias


class Softmax:
    """
    Computes softmax loss.
    This class simplifies the full operations of softmax.
    Reference to the full intution of softmax can be found here: http://cs231n.github.io/linear-classify/#softmax
    """

    def forwardPass(self, input1, label):
        """
        Forwards scores and the correct label through the function
        to evaluate it's loss
        """
        #saves input1 for use in calling scores
        self.input1 = input1

        #saves label input for backward pass
        self.labelIndex = label

        #prevents values from being too high to exponentiate (744)
        #Also tries not to shrink them so low they vanish to 0 and
        #cannot be logged.
        maxVal = np.max(input1)
        if(maxVal == 0): #prevents divide by zero error
            maxVal = 1/744
        expNum = input1 / (maxVal * 744)

        #exponentiates safe values
        exp = np.exp(expNum)

        #sums all weights and creates 1/sum to multiply
        #and find the probability scores
        sumVal = np.sum(exp)
        invSum = 1/sumVal

        #calculates probScores and saves for back pass
        #(scores are 0-1 probability based on networks scores)
        self._probScores = exp * invSum

        #computes loss (-log is 0 when score is 1, increases as score gets lower)
        self._loss = -math.log(self._probScores[self.labelIndex])


    def backwardPass(self, priorGradient = None):
        """
        Returns the gradient of the loss.
        There is never a priorGradient to the loss function,
        so it can be ignored
        """
        try:
            grad = self._probScores
            grad[self.labelIndex] -= 1
            return grad

        except NameError:
            print('Cannot backwardPass Softmax w/o a forward pass done first.')

    @property
    def loss(self):
        """
        Returns the loss of the network
        """
        try:
            return self._loss
        except NameError:
            print('No Loss value has been created.\nNeed to perform a forward pass to get loss')

    @property
    def scores(self):
        return self.input1

    @property
    def probScores(self):
        return self._probScores