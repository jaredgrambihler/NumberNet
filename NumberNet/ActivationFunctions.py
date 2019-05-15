import numpy as np

class ReLU:
    """
    ReLU acitivation layer
    Must take in a 1D vector
    """

    def forwardPass(self, input1):
        """
        Forces negative values to 0
        Returns result
        """
        self.result = np.maximum(0, input1)
        return self.result


    def backwardPass(self, priorGradient):
        """
        Sets not-activated values to 0 for the gradient as well.
        Returns the new gradient.
        """

        #sets return array values to 1, same size as priorGradient
        multGrad = np.ones(priorGradient.shape)

        #if result was 0, set gradient to 0
        for i in range(len(multGrad)):
            if(self.result[i] == 0):
                multGrad[i] = 0
        
        #all values are preserved *1 or forced to 0
        return multGrad * priorGradient


class LeakyReLU:

    def forwardPass(self, input1):
        self.input1 = input1
        result = input1
        for i in range(len(result)):
            if result[i] < 0:
                result[i] *= .1
        return result
        

    def backwardPass(self, priorGradient):
        grad = priorGradient
        for i in range(len(priorGradient)):
            if(self.input1[i] < 0):
                grad[i] *= .1
        return grad


class Sigmoid:
    """
    1/(1+ e^-x) activation function
    """
    def forwardPass(self, input1):
        """
        forwards an input vector through sigmoid function.
        May encounter an overflow value in runtime from very
        negative inputs leading the activation to 0.
        """
        self.input1 = input1
        min = np.amin(input1)
        result = 1 / (1 + np.exp(-input1))
        return result


    def backwardPass(self, priorGradient):
        """
        Takes the derivative of sigmoid and mutliplies it by the prior gradient.
        """
        localGrad = 1 / (2 + np.exp(self.input1) + np.exp(-self.input1))
        return localGrad * priorGradient