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
    #takes an optional distribution size
    def __init__(self, x, y, distribution = .01):

        #guassian distribution w/ sd of sqrt(2/inputs)
        self.weights = np.random.randn(x*y) * math.sqrt(2.0/y)
        self.weights = np.reshape(self.weights, (x,y))


    #backwardPass for gradients on weights
    #this method is irrelevant for a weights matrix
    def backwardPass(self, priorGradient):
        pass


    #updates the weight matrix based on a gradient and stepsize
    #this method should be called after a miniBatch computes a gradient.
    def updateGrad(self, stepSize, grad, regularization = .3):

        #performs update
        self.weights -= grad * stepSize

        #regularization function of L2 Regularization
        #Reference : http://cs231n.github.io/neural-networks-2/
        self.weights -= regularization * self.weights



#Addition layer for biases
#could be built to work in weights multiplication (simplify code)
class Bias(Layer):

    #inits biases to be 0
    def __init__(self, x):
        self.bias = np.zeros(x)

    
    #adds two vectors
    def forwardPass(self, input1):
       return input1 + self.bias
    

    #passes back gradient
    def backwardPass(self, priorGradient):
        return priorGradient

    #updates biases
    def updateGrad(self, stepSize, grad):
        self.bias -= stepSize * grad



#dot product b/w two vectors.
#Code can likely be simplified to be more efficient
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
        grad1 = np.zeros(self.input1.shape)

        #loops through number of previous gradients
        for i in range(len(priorGradient)):

            #loops through all previous inputs of 1d vector
            for j in range(len(self.input2)):

                #adds to the weightGrad the priorGradient * 1d value for that element
                #Creates a weightGrad matrix with all derivatives
                grad1[i][j] = priorGradient[i] * self.input2[j]

        grad2 = np.zeros(len(self.input1[0]))

        for i in range(len(self.input1[0])):
            sum = 0

            for j in range(len(self.input1)):
                sum += self.input1[j][i] * self.input2[i]
                sum *= priorGradient[j]

            grad2[i] += sum

        return grad1, grad2



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
        
        #all values are preserved *1 or forced to 0
        return multgrad * priorGradient



#Computes a softmax loss.
#the reference to this function is here:
#http://cs231n.github.io/linear-classify/#softmax
class Softmax(Layer):

    #forwards scores and the correct input through the function
    def forwardPass(self, input1, label):

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
        self.probScores = exp * invSum

        #computes loss
        self.loss = -math.log(self.probScores[self.labelIndex])


    #simplified derived gradient for softmax loss.
    #the whole function isn't showed in this math.
    def backwardPass(self, priorGradient = 1.00):

        grad = self.probScores
        grad[self.labelIndex] -= 1
            
        return grad



###############################################
#The following methods weren't found to be    #
#useful for creating networks, but could have #
#some use to a unique classifier, so they are #
#left below.                                  #
###############################################


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



#class for subtractice matrices.
#not sure if this is ever needed.
#HAS NOT BEEN TESTED
class Subtraction(Layer):

    #first input vector is the first in operation.
    def forwardPass(self, input1, input2):
        return input1-input2


    #flips the gradient for subtracted element, passes back gradient for first element. 
    def backwardPass(self, priorGradient):
        return priorGradient, priorGradient*-1



#computes a scalar operation
#not sure if this is ever needed.
class Scalar(Layer):
    
    #takes a vector input and a scalar float value
    def forwardPass(self, input1, scalar):
        #saves scalar for backprop
        self.scalar = scalar
        return scalar * input1


    #returns scaled vector gradient
    def backwardPass(self, priorGradient):
        return self.scalar * priorGradient




#unique last layer to be used to handle the loss.
#Probably not necesseray.
class Loss(Layer):

    #doesn't need to pass forward.
    def forwardPass(self, input1):
        self.loss = input1
        #outputs data?


    #gradient of the loss w/ respect to itself is always 1.
    def backwardPass(self):
        return 1.00
    


