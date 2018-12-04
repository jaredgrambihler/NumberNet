import numpy as np
import math


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


    #fixes error for 2 gradients combining in backprop
    def backwardPass(self, gradient1, gradient2):
        gradient = gradient1 + gradient2
        backwardPass(gradient)



#class for weights
class Weights:

    #random init of weights for x,y dimensions
    #takes an optional distribution size for determining
    #the spread of the normal distribution
    def __init__(self, x, y, distribution = .01):
        self.weights = np.random.randn(x, y) * distribution


    #backwardPass for gradients on weights
    #this method is irrelevant for a weights matrix
    def backwardPass(self, priorGradient):
        pass


    #updates the weight matrix based on a gradient and stepsize
    def updateGrad(self, stepSize, grad):
        self.weights -= grad * stepSize
        #takes sum using abs values
        weightSum = sum(sum(abs(self.weights)))
        #prevents weights from exploding using the sum, should be fixed later
        #self.weights = self.weights / (weightSum * 784000)
        #prevents weights from exploding
        if(weightSum > 80):
            self.weights /= weightSum



#unique last layer to be used to handle the loss.
#might not be necesseray, depending on how output
#data is coded.
class Loss(Layer):

    #doesn't need to pass forward.
    def forwardPass(self, input1):
        self.loss = input1
        #outputs data


    #gradient of the loss w/ respect to itself is always 1.
    def backwardPass(self):
        return 1.00
    


#adds too matrices together. Useful for biases.
#HAS NOT BEEN TESTED
class Addition(Layer):

    #adds two vectors
    def forwardPass(self, input1, input2):
       return input1 + input2
    

    #passes back gradient
    def backwardPass(self, priorGradient):
        return priorGradient



#class for subtractice matrices.
#HAS NOT BEEN TESTED
class Subtraction(Layer):

    #first input vector is the first in operation.
    def forwardPass(self, input1, input2):
        return input1-input2


    #flips the gradient for subtracted element, passes back gradient for first element. 
    def backwardPass(self, priorGradient):
        return priorGradient, priorGradient*-1



#computes a scalar operation
#HAS NOT BEEN TESTED
class Scalar(Layer):
    
    #takes a vector input and a scalar float value
    def forwardPass(self, input1, scalar):
        #saves scalar for backprop
        self.scalar = scalar
        return scalar * input1


    #returns scaled vector gradient
    def backwardPass(self, priorGradient):
        return self.scalar * priorGradient



#dot product b/w two vectors. Currently won't work if
#the input2 is a 2D vector. input1 is weights, input2
#is the image/previous layer. Need to be implemented
#to work with more than 1D input2 vectors.
class Multiplication(Layer):
    
    #Returns the dot product
    def forwardPass(self, input1, input2):
        #stores inputs, gets the weights matrix from weight layer passed in.
        self.input1 = input1.weights
        self.input2 = input2
        return self.input1.dot(self.input2)


    #returns grad only for input1
    def backwardPass(self, priorGradient):
        #creates an array for the gradients that is the same shape as the weights
        weightGrad = np.zeros(self.input1.shape)
        #loops through number of previous gradients
        for i in range(len(priorGradient)):
            #loops through all previous inputs of 1d vector
            for j in range(len(self.input2)):
                #adds to the weightGrad the priorGradient * 1d value for that element
                #Creates a weightGrad matrix with all derivatives
                weightGrad[i][j] = priorGradient[i] * self.input2[j]
        return weightGrad



#1/x division
#HAS NOT BEEN TESTED
class Division(Layer):

    #stores input1 and returns 1/input1
    def forwardPass(self, input1):
        self.input1 = input1
        return 1/input1


    #returns prior gradient times -1/x^2
    def backwardPass(self, priorGradient):
        return priorGradient * -1/(self.input1**2)



#computes max as a sum of the total array
#HAS NOT BEEN TESTED
class Max(Layer):

    #returns whatever array has a greater sum
    def forwardPass(self, input1, input2):
        #sums both arrays
        if(sum(input1) > sum(input2)):
            self.maxVal = True #these true/false refer to 1/2 input being true. This saves computation in the backwards pass.
            return input1
        else:
            self.maxVal = False
            return input2


    #routes gradient to max value
    #returns a 0 or the gradient matrix.
    def backwardPass(self, priorGradient):
        if(self.maxVal):
            return priorGradient, 0
        else:
            return 0, priorGradient



#natural log
#HAS NOT BEEN TESTED
class Log(Layer):

    #returns ln(x), stores input 
    def forwardPass(self, input1):
        self.input1 = input1
        return np.log(input1)

    #returns 1/x *grad
    def backwardPass(self, priorGradient):
        return priorGradient * 1/self.input1



#e^x
#HAS NOT BEEN TESTED
class Exp(Layer):

    #exponentiates the input
    def forwardPass(self, input1):
        #since derivative is equal to e^x, stores it to save compute
        self.backwardPass = np.exp(input1)
        return self.backwardPass


    #returns grad * previously computed e^x value
    def backwardPass(self, priorGradient):
        return self.backwardPass * priorGradient



#ReLU activation function
#HAS NOT BEEN TESTED
class ReLU(Layer):

    #forces negative values to 0
    def forwardPass(self, input1):
        self.result = np.maximum(0, input1)
        return self.result


    #sets not activated values to 0 for the gradient
    def backwardPass(self, priorGradient):
        #where self.result is 0, grad goes to 0
        #sets return array values to 1, same size as priorGradient
        multgrad = np.ones(priorGradient.shape)
        #forces 0 values to 0 for return gradient
        multgrad *= self.result 
        #all values are preserves *1 or forced to 0
        return multgrad * priorGradient



##computes the Softmax loss
#class Softmax(Layer):

#    #computes and returns a softmax loss
#    def forwardPass(self, input1, labelIndex):
#        #saves index for backwardPass
#        self.labelIndex = labelIndex
#        #exponentiates, can easily overflow (exploding gradients)
#        self.expValue = np.exp(input1)
#        #takes sum to normalize
#        self.sum = np.sum(self.expValue)
#        #saves compute by skipping to label index, ignores values that will go to 0
#        #this might have to be changed to preserve backwards gradient flow
#        self.normalizedValue = self.expValue[labelIndex] / self.sum
#        #computes loss as -log of the normalized values for the label score
#        self.loss = -math.log(self.normalizedValue)
#        return self.loss
    

#    #returns loss of softmax function (only one element is used since label is one element)
#    #priorGradeint is 1.00 by default (dL/dL should always be 1.00), but can be modified
#    #if needed.
#    def backwardPass(self, priorGradient = 1.00):
#        #finds gradient for the ln(x) portion
#        grad = priorGradient * -1/self.normalizedValue
#        #updates gradient to scale based on the sum
#        grad /= self.sum
#        #updates grad for e^x
#        grad *= self.expValue[self.labelIndex]
#         #makes a zeros array for return
#         #MAY BE PROBLEMATIC
#        returnGrad = np.zeros(self.expValue.shape)
#        #set the value of the gradient for the single gradient
#        #once again, may be problematic
#        returnGrad[self.labelIndex] = grad
#        return returnGrad



#computes the Softmax loss(test)
class Softmax(Layer):

    #computes and returns a softmax loss
    def forwardPass(self, input1, labelIndex):
        #saves index for backwardPass
        self.labelIndex = labelIndex
        #exponentiates, can easily overflow (exploding gradients)
        self.expValue = np.exp(input1)
        #takes sum to normalize
        self.sum = np.sum(self.expValue)
        #normalizes values by sum to give scores 0-1.0, all scores sum to 1.0
        self.normalizedValue = self.expValue / self.sum
        #computes loss as -log of the normalized values for the label score (1.0 score gives 0 loss)
        self.loss = -math.log(self.normalizedValue[self.labelIndex])
        return self.loss
    

    #returns loss of softmax function (only one element is used since label is one element)
    #priorGradeint is 1.00 by default (dL/dL should always be 1.00), but can be modified
    #if needed.
    def backwardPass(self, priorGradient = 1.00):
        #finds gradient for the ln(x) portion
        grad = priorGradient * -1/self.normalizedValue[self.labelIndex]
        #updates gradient to scale based on the sum
        grad /= self.sum
        #updates grad for e^x
        grad *= self.expValue
        #returns gradient
        return grad