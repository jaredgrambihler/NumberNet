import numpy as np
import math


class Layer:
    """
    Layer class to be implemented for different layer operations.
    All layers have a forward and backwardPass method.
    """
    
    def __init__(self):
        """
        Defines the general init to be a pass.
        Variables for layers are saved on a case by case basis in the 
        forward pass functions.
        """
        pass


    def forwardPass(self):
        """
        This method is unique for each layer.
        The information needed for the backward pass is saved to the class
        during the function.
        """
        pass


    def backwardPass(self, priorGradient):
        """
        Unique to each layer.
        Recieves a gradient for it's input
        return the local gradient times the prior gradient (chain rule)
        If several inputs exist to the function, they should be returned
        in the same order as the inputs in the forwardPass
        """
        pass


    def backwardPass(self, gradient1, gradient2):
        """
        Solves the potential problem of gradients coming
        together by adding them together
        """
        gradient = gradient1 + gradient2
        backwardPass(gradient)



class Weights:
    """
    Class to handle weights in the network
    """

    def __init__(self, x, y):
        """
        Creates weights based on the input size.
        x = number of outputs
        y = number of inputs
        """
        #guassian distribution w/ sd of sqrt(2/inputs)
        self._weights = np.random.randn(x*y) * math.sqrt(2.0/y)
        self._weights = np.reshape(self._weights, (x,y))

        #learning params
        self.vdw = np.zeros((x,y))
        self.sdw = np.zeros((x,y))


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



class Bias(Layer):
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



class Multiplication(Layer):
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




def main():
    def ReLUTest():
        relu = ReLU()
        input = np.array([-1,3,4,0,-.1,-10])
        print(relu.forwardPass(input))
        grad = np.array([x for x in range(len(input))])
        print(relu.backwardPass(grad))

    def leakyReLUTest():
        relu = LeakyReLU()
        input = np.array([-1,3,4,0,-.1,-10])
        print(relu.forwardPass(input))
        grad = np.array([float(x) for x in range(len(input))])
        print(relu.backwardPass(grad))

    def multTest():
        mult = Multiplication()
        weights = Weights(2,2)
        weights.weights[0][0] = .1
        weights.weights[0][1] = .5
        weights.weights[1][0] = -.3
        weights.weights[1][1] = .8
        print(weights.getWeights())
        input = np.array([.2,.4])

        print(mult.forwardPass(weights,input))
        print(mult.backwardPass(np.array([.44,.52])))

    print('Multiplication Layer Test:')
    multTest()

    print('ReLU Test:')
    ReLUTest()

    print('Leaky ReLU Test:')
    leakyReLUTest()

if __name__ == '__main__':
    main()
