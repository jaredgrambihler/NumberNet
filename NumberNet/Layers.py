"""
Contains classes relating to layers of the network.
Classes:
    Layer
    Bias
    Softmax
"""
import numpy as np
import math
from . import ActivationFunctions

class Layer:
    """
    Class to be defined for each layer of the network
    """
    #---Layer format---
    #Weights
    #Biases (could be part of weights for optimization)
    #Activation Function (can be None)

    def __init__(self, inputSize, outputSize, bias = False, activationFunction = None):
        """
        Initialize weights.
        Args:
            inputSize (int): Size of input to the layer
            outputSize (int): Size the layer output should be
            biase (boolean) (default = False): Determines if the layer has a bias. True indicates bias is present.
            activationFunction (String) (default = None): Activation function to be used. Can be 'ReLU', 'Sigmoid',
                or 'LeakyReLU' ('HipsterReLU')
        """
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

  
    def forwardPass(self, inputVector):
        """
        Completes a forward pass of a vector through the current Layer.
        Args:
            inputVector (numpy array): 1D numpy array that is the input to the layer. Should be of the same
                length as specified in inputSize.
        Returns:
            numpy array (1D). Dot product between input and weights
        Returns a dot product between them
        """
        #weights/bias classes handles actual computation forward pass
        #weights/bias class also store the information for the backwards pass.
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
        Computes gradients of weights/bias and returns gradient prior to this layer.
        Args:
            piorGradient (numpy array): 1D numpy array that is the gradient from the previous layer.
                Should be the same size as outputSize.
        Returns:
            numpy array. Length is the same as inputSize, and it is the gradient to be passed to the previous layer.
        """
        if(self.activationFunction):
            priorGradient = self.activationFunction.backwardPass(priorGradient)
        if(self.bias):
            priorGradient = self.bias.backwardPass(priorGradient)
        priorGradient = self.weights.backwardPass(priorGradient)
        return priorGradient

    def updateGrad(self, numData, parameters):
        """
        Updates weights and biases based on parameters
        Args:
            numData (int): number of samples that have been forwarded through the network prior to an update.
                Typically the minibatch size.
            parameters (Parameters): parameters object containing network parameters
        """
        self.weights.updateGrad(numData, parameters)
        self.bias.updateGrad(numData, parameters)


class Weights:
    """
    Class for weights on each layer
    """

    def __init__(self, inputSize, outputSize):
        """
        Creates randomized weights based on the input size.
        Args:
            inputSize (int): size of input to layer
            outputSize (int): size of output from layer
        """
        #guassian distribution w/ StdDev of sqrt(2/inputs)
        self._weights = np.random.randn(inputSize*outputSize) * math.sqrt(2.0/inputSize)
        self._weights = np.reshape(self._weights, (outputSize, inputSize))
        #learning params
        self.vdw = np.zeros((outputSize, inputSize))
        self.sdw = np.zeros((outputSize, inputSize))
        self.grad = np.zeros((outputSize, inputSize))


    def forwardPass(self, inputVector):
        """
        Forward input through weights.
        Args:
            inputVector (numpy array): 1D numpy array that is the same length as inputSize
        Returns:
            numpy array (1D) of length outputSize. Dot product between input and weights
        """
        self.input = inputVector #save for backward pass computation
        return np.dot(self._weights, inputVector)


    def backwardPass(self, prior):
        """
        Computes gradients of weights and returns gradient prior to this layer.
        Args:
            piorGradient (numpy array): 1D numpy array that is the gradient from the previous layer.
                Should be the same size as outputSize.
        Returns:
            numpy array. Length is the same as inputSize. It is the gradient prior to the weights computation.
        """
        self.input = np.array(self.input)
        #Compute gradient for weights
        if(len(self.input.shape) == 1):
            self.grad += np.dot(np.reshape(prior, (prior.shape[0], 1)),
                                np.reshape(self.input, (1, self.input.shape[0])))
        else:
            self.grad += np.dot(prior, np.reshape(self.input, (self.input.shape[1], self.input.shape[0])))
        return np.dot(np.transpose(self._weights), prior) #compute and return prior gradient


    def updateGrad(self, numData, parameters):
        """
        Updates the weights based on gradient, stepSize, and Regularization.
        Should be called after the avg gradient is computed for a minibatch
        Args:
            numData (int): number of vectors forwarded through the network before an update.
            parameters (Parameters): Parameters object. These parameters determine how the network updates.
        """
        #momentum and RMSProp implemented w/o correction
        self.grad /= numData    #Scale gradient values
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
        Numpy array of shape (outputSize, inputSize)
        """
        return self._weights


class Bias:
    """
    Class to hold biases in each Layer
    """

    def __init__(self, x):
        """
        Creates biases (set to all 1's)
        Args:
            x (int): size of biases
        """
        self._bias = np.ones(x)
        #learning params
        self.vdb = np.zeros(x)
        self.sdb = np.zeros(x)
        self.grad = np.zeros(x)


    def backwardPass(self, priorGradient):
        """
        Makes no change to gradient but needed to save gradient for updating bias
        Args:
            priorGradient (numpy array): 1D Numpy array of same length as biases
        Returns:
            numpy array, unchanged priorGradient.
        """
        self.grad += priorGradient #update stored gradient.
        return priorGradient


    def updateGrad(self, numData, parameters):
        """
        Performs an update to the biases.
        Args:
            numData (int): number of vectors forwarded through bias prior to updating
            parameters (Parameters): network parameters to specify how to update.
        """
        self.grad /= numData    #scale gradient
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
        1D Numpy array
        """
        return self._bias


class Softmax:
    """
    Computes softmax loss.
    This class simplifies the full operations of softmax.
    """
    #Reference to the full intution of softmax can be found here: http://cs231n.github.io/linear-classify/#softmax


    def forwardPass(self, input1, label):
        """
        Forwards scores and the correct label through the function to evaluate it's loss
        Args:
            input1 (numpy array): 1D numpy array. Output from last layer of network.
            label (int): int corresponding to label of current input
        """
        self.input1 = input1     #saves input1 for use in calling scores
        self.labelIndex = label  #saves label input for backward pass
        #prevents values from being too high to exponentiate (cannot be >744)
        #Also tries not to shrink them so low they vanish to 0 and cannot be logged.
        maxVal = np.max(input1)
        if(maxVal == 0):    #prevents divide by zero error
            maxVal = 1/744
        expNum = input1 / (maxVal * 744)
        exp = np.exp(expNum) #exponentiates safe values
        #sums all weights and creates 1/sum to multiply and find the probability scores
        sumVal = np.sum(exp)
        invSum = 1/sumVal
        #calculates probScores and saves for back pass (scores are 0-1 probability based on networks scores)
        self._probScores = exp * invSum
        #computes loss (-log is 0 when score is 1, increases as score gets lower)
        self._loss = -math.log(self._probScores[self.labelIndex]) 


    def backwardPass(self):
        """
        Returns:
            Gradient prior to loss fuinction
        """
        #Doesn't need priorGradient because there's nothing prior to loss function
        try:
            grad = self._probScores
            grad[self.labelIndex] -= 1
            return grad
        except NameError:
            print('Cannot backwardPass Softmax w/o a forward pass done first.')

    @property
    def loss(self):
        """
        Float, loss of network. 0 represents perfect accuracy, higher values represent greater loss.
        """
        try:
            return self._loss
        except NameError:
            print('No Loss value has been created.\nNeed to perform a forward pass to get loss')

    @property
    def scores(self):
        """
        Numpy array. Output from network before probability scores are computed.
        """
        return self.input1

    @property
    def probScores(self):
        """
        Numpy array. Probability scores from the network (0-1 for each value's probability)
        """
        return self._probScores
