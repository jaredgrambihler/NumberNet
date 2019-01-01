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
        self.weights = np.random.randn(x*y) * math.sqrt(2.0/y)
        self.weights = np.reshape(self.weights, (x,y))


    def updateGrad(self, stepSize, grad, regularization):
        """
        Updates the weights based on gradient, stepSize, and Regularization.
        Should be called after the avg gradient is computed for a minibatch
        """

        #performs update
        self.weights -= grad * stepSize

        #regularization function of L2 Regularization
        #Reference : http://cs231n.github.io/neural-networks-2/
        self.weights -= regularization * self.weights

    def getWeights(self):
        """
        Returns the weights
        """
        return self.weights



class Bias(Layer):
    """
    Layer for biases.
    Should be implemented after weights and input are multiplied together.
    """

    def __init__(self, x):
        """
        Creates biases of length 'n' to all be zero
        """
        self.bias = np.zeros(x)


    def forwardPass(self, input1):
        """
        adds the biases to the 1D input vector
        Returns Result
        """
        return input1 + self.bias
    

    def backwardPass(self, priorGradient):
        """
        There is no local gradient to an addition function.
        Returns the priorGradient as is
        """
        return priorGradient


    def updateGrad(self, stepSize, grad):
        """
        Performs an update to the biases.
        Should be done after the avg is computes from a minibatch
        """
        self.bias -= stepSize * grad

    def getBias(self):
        """
        Returns the biases
        """
        return self.bias



class Multiplication(Layer):
    """
    Multiplies together two matrices
    weights = weights matrix
    input = 1D input vector
    """

    def forwardPass(self, weights, input):
        """
        stores weights and input for the backward pass.
         Returns a dot product between them
        """
        self.weights = weights.getWeights()
        self.input = input
        return np.dot(self.weights, self.input)


    def backwardPass(self, priorGradient):
        """
        Computes gradients of both the weights and the input
        Returns weightGrad, inputGrad
        """
        #creates matrices of the proper size to hold the gradients
        weightGrad = np.ones(np.shape(self.weights))
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
            weightRow = self.weights[:, i]
            #multiplies together the row each priorGradient that it was related t0
            tempGrad = np.multiply(weightRow, priorGradient)
            #takes the sum of the 'i' row impact as the local gradient 
            #(already multiplied to the prior gradient)
            inputGrad[i] = np.sum(tempGrad)

        return weightGrad, inputGrad



class ReLU(Layer):
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
        return multgrad * priorGradient



class Softmax(Layer):
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
        expNum = input1 / (maxVal * 744)

        #exponentiates safe values
        exp = np.exp(expNum)

        #sums all weights and creates 1/sum to multiply
        #and find the probability scores
        sumVal = np.sum(exp)
        invSum = 1/sumVal

        #calculates probScores and saves for back pass
        #(scores are 0-1 probability based on networks scores)
        self.probScores = exp * invSum

        #computes loss (-log is 0 when score is 1, increases as score gets lower)
        self.loss = -math.log(self.probScores[self.labelIndex])


    def backwardPass(self, priorGradient = None):
        """
        Returns the gradient of the loss.
        There is never a priorGradient to the loss function,
        so it can be ignored
        """
        try:
            grad = self.probScores
            grad[self.labelIndex] -= 1
            return grad

        except NameError:
            print('Cannot backwardPass Softmax w/o a forward pass done first.')


    def getLoss(self):
        """
        Returns the loss of the network
        """
        try:
            return self.loss
        except NameError:
            print('No Loss value has been created.\nNeed to perform a forward pass to get loss')


    def getScores(self):
        """
        Returns the scores for the network before probScores
        """
        return self.input1

    


