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