"""
Contains classes for all Activation Functions that are used.
"""
import numpy as np

class ReLU:
    """
    ReLU acitivation layer.
    Methods:
        forwardPass
        backwardPass
    """

    def __init__(self):
        self._forwardComplete = False #set to True after forward pass is run
                                      #ensures backward pass can't run without data. 

    def forwardPass(self, input1):
        """
        Applies ReLU (forcing negative values to 0) to a vector.
        Input is remembered for gradient computation in backward pass.
        Args:
            input1 (numpy array): 1D input vector
        Returns:
            numpy array: input1 with ReLU applied
        """
        self._forwardComplete = True
        self.result = np.maximum(0, input1)
        return self.result


    def backwardPass(self, priorGradient):
        """
        Backward pass for ReLU.
        Pre:
            forwardPass has been run on this object.
        Args:
            priorGradient (numpyArray): 1D array of prior gradient from the next layer in the network.
                Assumed to be the same dimensions as the input was.
        Returns:
            numpy Array: gradient prior to this layer
        """
        if(not self._forwardComplete):
            raise Exception("Cannot backwards pass an image without a forward pass first!")
        multGrad = np.ones(priorGradient.shape) #sets return array values to 1, same size as priorGradient
        #if result of ReLU was 0, set gradient to 0 at that location
        for i in range(len(multGrad)):
            if(self.result[i] == 0):
                multGrad[i] = 0
        return multGrad * priorGradient #all values are preserved by *1 or forced to 0


class LeakyReLU:
    """
    LeakyReLU Activation layer.
    Methods:
        forwardPass
        backwardPass
    """

    def forwardPass(self, input1):
        """
        Applies LeakyReLU to a layer.
        Input is remembered for gradient computation in backward pass.
        Args:
            input1 (numpy array): 1D input vector
        Returns:
            numpy array: input1 with LeakyRelu applied
        """
        self.input1 = input1
        result = input1
        for i in range(len(result)):
            if result[i] < 0:
                result[i] *= .1
        return result
        

    def backwardPass(self, priorGradient):
        """
        Backward pass for LeakyReLU.
        Pre:
            forwardPass has been run on this object.
        Args:
            priorGradient (numpyArray): 1D array of prior gradient from the next layer in the network.
                Assumed to be the same dimensions as the input was.
        Returns:
            numpy Array: gradient prior to this layer
        """
        grad = priorGradient
        for i in range(len(priorGradient)):
            if(self.input1[i] < 0):
                grad[i] *= .1
        return grad


class Sigmoid:
    """
    Sigmoid activation layer.
    1/(1+ e^-x) activation function
    Methods:
        forwardPass
        backwardPass
    """

    def forwardPass(self, input1):
        """
        Applies LeakyReLU to a layer.
        Input is remembered for gradient computation in backward pass.
        May encounter an overflow value in runtime from very
        negative inputs leading the activation to 0.
        Args:
            input1 (numpy array): 1D input vector
        Returns:
            numpy array: input1 with Sigmoid applied
        """
        self.input1 = input1
        min = np.amin(input1)
        result = 1 / (1 + np.exp(-input1))
        return result


    def backwardPass(self, priorGradient):
        """
        Backward pass for Sigmoid.
        Pre:
            forwardPass has been run on this object.
        Args:
            priorGradient (numpyArray): 1D array of prior gradient from the next layer in the network.
                Assumed to be the same dimensions as the input was.
        Returns:
            numpy Array: gradient prior to this layer
        """
        #Takes the derivative of sigmoid and mutliplies it by the prior gradient.
        localGrad = 1 / (2 + np.exp(self.input1) + np.exp(-self.input1))
        return localGrad * priorGradient