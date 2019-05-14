import numpy as np
import math
import ActivationFunctions

class Layer:
    """
    Class to be defined for each layer of the network.
    """
    #Layer format
    #input - (maybe?) allocate to store forward pass input
    #Weights
    #Biases (can move to weights later to optimize)
    #Activation Function (can be None)

    def __init__(self, inputSize, outputSize, bias = False, activationFunction = None):
        #make inputs take parameters?
        self.input = np.array(inputSize, outputSize)
        self.weigths = []
        self.bias = []
        #Sets activation function
        if(activationFunction):
            if activationFunction == 'ReLU':
                self.activationFunction = ActivationFunctions.ReLU()
            elif activationFunction == 'Sigmoid':
                self.activationFunction = ActivationFunctions.Sigmoid()
            elif activationFunction == 'LeakyReLU' or activationFunction == 'HipsterReLU':
                self.activationFunction = ActivationFunctions.LeakyReLU()
        else:
            self.actiavtionFuntion = None

    #FORWARD AND BACKWARD PASS FROM MULTIPLICATION LAYER
    def forwardPass(self, weights, inputVector):
        """
        stores weights and input for the backward pass.
        Returns a dot product between them
        """
        self._weights = weights.weights
        self.input = inputVector
        return np.dot(self._weights, self.input)

    def backwardPass(self, priorGradient):
        """
        Computes gradients of both the weights and the input
        Returns weightGrad, inputGrad
        """
        #creates matrices of the proper size to hold the gradients
        weightGrad = np.ones(np.shape(self._weights))
        inputGrad = np.ones(np.shape(self.input))

        #creates the weightGrad
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

        return weightGrad, inputGrad


class Weights:
    """Class for weights on each layer"""
    def __init__(self, inputSize, outputSize):
        """
        Creates weights based on the input size.
        x = number of outputs
        y = number of inputs
        """
        #guassian distribution w/ sd of sqrt(2/inputs)
        self._weights = np.random.randn(inputSize*outputSize) * math.sqrt(2.0/outputSize)
        self._weights = np.reshape(self._weights, (inputSize,outputSize))

        #learning params
        self.vdw = np.zeros((inputSize,outputSize))
        self.sdw = np.zeros((inputSize,outputSize))


    def updateGrad(self, grad, parameters):
        """
        Updates the weights based on gradient, stepSize, and Regularization.
        Should be called after the avg gradient is computed for a minibatch
        """

        #momentum and RMSProp implemented w/o correction
        if parameters.momentum == True:
            self.vdw = parameters.beta1 * self.vdw + (1- parameters.beta1) * grad
            self._weights -= parameters.stepSize * self.vdw
        else:
            self._weights -= parameters.stepSize * grad

        if parameters.RMSProp == True:
            self.sdw = parameters.beta2 * self.sdw + (1 - parameters.beta2) * grad**2
            self._weights /= (self.sdw + parameters.epsilon)**.5

        #regularization function of L2 Regularization
        #Reference : http://cs231n.github.io/neural-networks-2/
        self._weights -= parameters.regularization * self.weights

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

    def __init__(self, x):
        """
        Creates biases of length 'n' to all be zero
        """
        self._bias = np.ones(x)
        self.vdb = np.zeros(x)
        self.sdb = np.zeros(x)



    def forwardPass(self, input1):
        """
        adds the biases to the 1D input vector
        Returns Result
        """
        return input1 + self._bias
    

    def backwardPass(self, priorGradient):
        """
        There is no local gradient to an addition function.
        Returns the priorGradient as is
        """
        return priorGradient


    def updateGrad(self, grad, parameters):
        """
        Performs an update to the biases.
        Should be done after the avg is computes from a minibatch
        """
        if parameters.momentum == True:
            self.vdb = parameters.beta1 * self.vdb + (1 - parameters.beta1) * grad
            self._bias -= self.vdb * parameters.stepSize
        else:
            self._bias -= grad * parameters.stepSize

        if parameters.RMSProp == True:
            self.sdb = parameters.beta2 * self.sdb + (1 - parameters.beta2) * grad**2
            self._bias /= (self.sdb + parameters.epsilon)**.5

    @property
    def bias(self):
        """
        Returns the biases
        """
        return self._bias



class Multiplication:
    """
    Multiplies together two matrices
    weights = weights matrix
    input = 1D input vector
    """

    def forwardPass(self, weights, inputVector):
        """
        stores weights and input for the backward pass.
        Returns a dot product between them
        """
        self._weights = weights.weights
        self.input = inputVector
        return np.dot(self._weights, self.input)


    def backwardPass(self, priorGradient):
        """
        Computes gradients of both the weights and the input
        Returns weightGrad, inputGrad
        """
        #creates matrices of the proper size to hold the gradients
        weightGrad = np.ones(np.shape(self._weights))
        inputGrad = np.ones(np.shape(self.input))

        #creates the weightGrad
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

        return weightGrad, inputGrad

    @property
    def weights(self):
        return self._weights