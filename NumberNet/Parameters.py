class Parameters():
    """
    Class used to pass hyperparameters for the network around
    using only one variable
    """

    def __init__(self, stepSize, regularization, decay, RMSProp = True, momentum = True):
        self._initialStepSize = stepSize
        self._stepSize = self._initialStepSize
        self._regularization = regularization
        self._decay = decay

        if momentum == True:
            self._momentum = True
            self._beta1 = .9
        else:
            self._momentum = False

        if RMSProp == True:
            self._RMSProp = True
            self._beta2 = .99
            self._epsilon = 1e-8
        else:
            self._RMSProp = False

    #getters for the class, avoids changing of variables
    @property
    def initialStepSize(self):
        return self._initialStepSize
   
    @property
    def stepSize(self):
        return self._stepSize

    @property
    def regularization(self):
        return self._regularization
    
    @property
    def decay(self):
        return self._decay

    @property
    def RMSProp(self):
        return self._RMSProp

    @property
    def momentum(self):
        return self._momentum

    @property
    def beta1(self):
        return self._beta1

    @property
    def beta2(self):
        return self._beta2

    @property
    def epsilon(self):
        return self._epsilon

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize