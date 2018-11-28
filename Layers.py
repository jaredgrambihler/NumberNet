import numpy as np

class Layer:
    
    #defines init for all methods to be a pass. Variables should be created
    #during the forward and backward passes. Multiple input operations (e.g. 5+3+2) 
    #should be computed using two seperate gate combinations (e.g. do the 5+3 operation, 
    #then do +2 to that answer) to manage variables in layers.
    def __init__(self):
        pass


    #to be implemented for each layer. Returns an output to be passed forward
    #and stored in the following layer.
    def forwardPass(self):
        pass

    #to be implemented for each layer. Recieves a gradient as input, returns
    #a gradient to be passed back to prior layer. If there are several inputs,
    #the values are returned in order of the input #'s
    def backwardPass(self, priorGradient):
        pass

    #fixes error for 2 gradients combining.
    def backwardPass(self, gradient1, gradient2):
        gradient = gradient1 + gradient2
        backwardPass(gradient)


#class for weights
class Weights:

    #random init of weights for x,y dimensions
    def __init__(x, y):
        self.weights = np.random.random((x, y))

    #backwardPass for gradients on weights
    def backwardPass(self, priorGradient):
        pass

    #updates the weight gradient based on avg of minibatch
    def updateGrad(self, stepSize, avgGrad):
        self.weights -= avgGrad *stepSize


#unique last layer to be used to handle the loss.
class Loss(Layer):

    #doesn't need to pass forward.
    def forwardPass(self, input1):
        self.loss = input1
        #outputs data

    #gradient of the loss w/ respect to itself is always 1.
    def backwardPass(self):
        return 1.00
    

class Addition(Layer):

    #adds two vectors
    def forwardPass(self, input1, input2):
       return input1 + input2
    
    #passes back gradient
    def backwardPass(self, priorGradient):
        return priorGradient


class Subtraction(Layer):

    #first input vector is the first in operation.
    def forwardPass(self, input1, input2):
        return input1-input2

    #flips the gradient for subtracted element, passes back gradient for first element. 
    def backwardPass(self, priorGradient):
        return priorGradient, priorGradient*-1


#computes a scalar operation
class Scalar(Layer):
    
    #takes a vector input and a scalar float value
    def forwardPass(self, input1, scalar):
        self.scalar = scalar
        return scalar * input1

    #returns scaled vector gradient
    def backwardPass(self, priorGradient):
        return self.scalar * priorGradient


#dot product b/w two vectors
class Multiplication(Layer):
    
    #stores inputs and returns the dot product
    def forwardPass(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        return input1.dot(input2)

    #returns grad of input1 and input2
    def backwardPass(self, priorGradient):
        #scale vector for each row that would have been involved in dot product.
        #It is important that input1 is the larger vector and input2 is 1D!'
        returnArray = np.vstack((self.input1[0,:], self.input1[1,:]))
        for i in range(self.input1.shape[0]-2):
            returnArray = np.vstack((returnArray, self.input1[i+2,:]))
        return returnArray * priorGradient, self.input1.dot(priorGradient)


#1/x division
class Division(Layer):

    #stores input1 and returns 1/input1
    def forwardPass(self, input1):
        self.input1 = input1
        return 1/input1

    #returns prior gradient times -1/x^2
    def backwardPass(self, priorGradient):
        return priorGradient * -1/(self.input1**2)


#computes max as a sum of the total array
class Max(Layer):

    #returns whatever array has a greater sum
    def forwardPass(self, input1, input2):
        if(input1.sum() > input2.sum()):
            self.maxVal = True #these true/false refer to 1/2 input being true. This saves computation in the backwards pass.
            return input1
        else:
            self.maxVal = False
            return input2

    #routes gradient to max value
    def backwardPass(self, priorGradient):
        if(self.maxVal):
            return priorGradient, 0
        else:
            return 0, priorGradient


#natural log
class Log(Layer):

    #takes ln(x), stores input 
    def forwardPass(self, input1):
        self.input1 = input1
        return np.log(input1)

    #returns 1/x *grad
    def backwardPass(self, priorGradient):
        return priorGradient * 1/self.input1


#e^x
class Exp(Layer):

    #exponentiates the input
    def forwardPass(self, input1):
        self.backwardPass = np.exp(input1) #since derivative is equal to e^x, stores it to save compute
        return self.backwardPass

    #returns grad * previously computed e^x value
    def backwardPass(self, priorGradient):
        return self.backwardPass * priorGradient


class ReLU(Layer):

    #forces negative values to 0
    def forwardPass(self, input1):
        self.result = np.maximum(0, input1)
        return self.result

    #sets not activated values to 0 for the gradient
    def backwardPass(self, priorGradient):
        #where self.result is 0, grad goes to 0
        multgrad = np.ones(self.result.shape) #sets values to 1
        multgrad *= self.result #forces all 0 values to 0
        return multgrad * priorGradient #all values are preserves *1 or forced to 0


class SoftMax(Layer):

    #computes and returns a softmax loss
    def forwardPass(self, input1, labelIndex):
        self.labelIndex = labelIndex #saves index for backwardPass
        self.expValue = np.exp(input1) #exponentiates
        self.sum = np.sum(self.expValue) #takes sum
        self.normalizedValue = self.expValue[labelIndex] / sum #saves compute by skipping to label index
        self.loss = -log(self.normalizedValue)
        return self.loss

    #returns loss of softmax function (only one element is used since label is one element)
    def backwardPass(self, priorGradient):
        grad = priorGradient * -1/self.normalizedValue
        grad /= self.sum
        grad *= self.expValue[self.labelIndex]
        returnGrad = np.zeros(self.expValue.shape) #makes a zeros array for return
        returnGrad[labelIndex] = grad #set the value of the gradient
        return returnGrad
