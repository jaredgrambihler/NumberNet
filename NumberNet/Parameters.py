"""
Contains Parameters class
"""

class Parameters():
    """
    Class used to store hyperparameters for training the Network.
    """

    def __init__(self, stepSize, regularization, decay, RMSProp = True, momentum = True):
        """
        Initialize parameters:
        Args:
            stepSize (float): step size for training.
            regulatization (float): regularization strength for training
            decay (float): decay strength for training.
            RMSProp (boolean) (default: True): Whether or not to use RMSProp during trainig
            momentum (boolean) (default: True): Whether or not to use momentum when training
        """
        self._initialStepSize = stepSize        #base step size
        self._stepSize = self._initialStepSize  #current step size. May vary throughout training
        self._regularization = regularization   #regularization strength
        self._decay = decay                     #decay strength
        if momentum == True:
            self._momentum = True               #determines use of momentum
            self._beta1 = .9                    #sets beta for momentum
        else:
            self._momentum = False          
        if RMSProp == True:
            self._RMSProp = True                #determines use of RMSProp
            self._beta2 = .99                   #sets beta for RMSProp
            self._epsilon = 1e-8                #sets epsilon for RMSProp
        else:
            self._RMSProp = False


    #getters for the class
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