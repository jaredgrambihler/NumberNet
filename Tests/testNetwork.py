import unittest
from NumberNet import Network
from NumberNet.Parameters import Parameters
from NumberNet.Layers import Layer, Softmax
import numpy as np

class NetworkTests(unittest.TestCase):
    """
    Tests for Network class
    """

    def setUp(self):
        """
        WEIGHTS       BIAS  ACTIVATION    PARAMETERS
        .22 .24 .44   .2    Softmax       stepSize: .1
        .33 .42 .68   .3                  regularization: 1e-5
                                          decay: .9
        """
        params = Parameters(stepSize=.1, regularization=1e-5, decay=0, RMSProp=False, momentum=False)
        layers = [Layer(3,2,True)]
        lossFunc = Softmax()
        self.network = Network(parameters=params, layers=layers, lossFunction=lossFunc)
        #now set weights and bias to predefined values
        self.network.layers[0].weights._weights = np.array([[.22,.24,.44], [.33,.42,.68]])
        self.network.layers[0].bias._bias = np.array([.2,.3])
        self.input = np.array([.1,.2,.3])
        self.label = 1

    @unittest.skip("Unwritten Test")
    def testInit(self):
        """
        Test initialization of Network based on parameters, layers, lossFunction
        Test intialization with bad arguments
        """
        pass

    def testDataNormalization(self):
        trainImages = np.array([[.5,.5], [0,0]])
        self.network.setNormalizeData(trainImages)
        expectedMeanImg = np.array([.25,.25])
        for i, x in enumerate(expectedMeanImg):
            self.assertAlmostEqual(self.network.meanImg[i], x)
        #TODO - check on variance

    def testForwardPass(self):
        """
        WEIGHTS       INPUT BIAS OUPUT  ACTIVATION  OUTPUT (5 points acc)    
        .22 .24 .44   .1    .2    .402   Softmax     .44546
        .33 .42 .68   .2    .3--> .621           --> .55453
                      .3              
        """
        output = self.network.forwardPass(self.input, self.label)
        expectedOutput = np.array([.44546,.55453])
        for i, x in enumerate(expectedOutput):
            self.assertAlmostEqual(output[i], x, 4, msg="Error in forward pass")

    def testBackwardPass(self):
        """
        OUTPUT (5 points acc)   SOFTMAX GRAD     BIAS GRADIENT 
        .44546                  .44546          .44546
        .55453                 -.44546         -.44546
        
        WEIGHTS GRADIENT            OUTPUT GRADIENT
         .04454  .08909  .13363    -.04900
        -.04454 -.08909 -.13363    -.08018
                                   -.10691
        """
        self.network.forwardPass(self.input, self.label) #forward pass to set internal states
        self.network.backwardPass()
        #function doesn't return the final value but we can check internal states of everything
        expectedBiasGrad = np.array([.44546,-.44546])
        biasGrad = self.network.layers[0].bias.grad
        for i, x in enumerate(expectedBiasGrad):
            self.assertAlmostEqual(biasGrad[i], x, 4, msg="Error in bias gradient")
        expectedWeightsGrad = np.array([[.04454,.08909,.13363],[-.04454,-.08909,-.13363]]).reshape(6)
        weightsGrad = self.network.layers[0].weights.grad.reshape(6)
        for i, x in enumerate(expectedWeightsGrad):
            self.assertAlmostEqual(weightsGrad[i], x, 4, msg="Error in weigts gradient")
        #TODO - find a crafty way to check the output gradient

    def testAccuracy(self):
        """
        WEIGHTS       BIAS  ACTIVATION
        .22 .24 .44   .2    Softmax      
        .33 .42 .68   .3                  
        INPUT           Output         
        .1 .2 .3   -->  .44546  .55453                   
        """
        testImages = np.array([[.1,.2,.3], [.1,.2,.3], [.1,.2,.3]])
        testLabels = [0,1,1] #correct label is 1
        accuracy = self.network.accuracyTest(testImages, testLabels)
        self.assertAlmostEqual(2/3, accuracy, msg="Inaccurate accuracy")

    def testBatchTraining(self):
        """
        WEIGHTS GRADIENT            STEP SIZE ADJUSTMENT
         .04454  .08909  .13363     .004454  .008909  .013363
        -.04454 -.08909 -.13363    -.004454 -.008909 -.013363

        WEIGHTS                     ADJUSTED WEIGHTS
        .22 .24 .44                 .215546  .231091  .426637
        .33 .42 .68                 .334454  .428909  .693363
        """
        #set normalized data to not do anything to image so math is the same
        #as prior tests
        self.network.setNormalizeData(np.array([[0,0,0]]))
        #run back on input
        self.network.trainBatch(np.array([self.input]), np.array([self.label]), 0)
        #check to see if weights updated correctly
        expected = np.array([.215546, .231091, .426637, .334454, .428909, .693363])
        actual = self.network.layers[0].weights._weights.reshape(6)
        for i, x in enumerate(expected):
            self.assertAlmostEqual(x, actual[i], 4, msg="Improper weights update in train batch")