"""
Tests network functions!
"""
import unittest
import numpy as np
import copy
from NumberNet.ActivationFunctions import ReLU, LeakyReLU, Sigmoid
from NumberNet.Parameters import Parameters
from NumberNet.Layers import *

class LayerTests(unittest.TestCase):
    """
    Tests for Layer class
    """
    
    """
    Due to the nature of tests, there are separate setup functions for each type of layer to be tested
    this allows for choosing flexible inputs for each layer (e.g. ensuring ReLU would be activated) and
    sets self variables in a similar way to the traditional setUp method, we just have to ensure we call
    the appropriate setup for the given type of layer we want to test.
    """
    def setUpNoBiasOrActivation(self):
        self.input = np.array([.2,.4])
        self.priorGradient = np.array([.44,.52])
        self.layerNoBiasOrActivation = Layer(2,2)
        self.layerNoBiasOrActivation.weights._weights = np.array([[.1,.5],[-.3,.8]])
        self.expectedForwardPass = np.array([.22,.26])
        self.expectedBackwardPass = np.array([-.112,.636])
        self.expectedGradient = np.array([[.088,.176],[.104,.208]]).reshape(4)

    def setUpWithBiasNoActivation(self):
        self.setUpNoBiasOrActivation()
        self.layerWithBiasNoActivation = copy.deepcopy(self.layerNoBiasOrActivation)
        self.layerWithBiasNoActivation.bias = Bias(2)
        self.layerWithBiasNoActivation.bias._bias = np.array([.5,.2])
        #add bias to expected forward pass
        for i in range(len(self.expectedForwardPass)):
            self.expectedForwardPass[i] += self.layerWithBiasNoActivation.bias._bias[i]
        #expected backwards pass in unchanged from no bias
        #expected gradient is unchanged from no bias

    def setUpNoBiasReLU(self):
        self.setUpNoBiasOrActivation()
        self.input = np.array([-.8, .1])
        self.layerNoBiasReLU = copy.deepcopy(self.layerNoBiasOrActivation)
        self.layerNoBiasReLU.activationFunction = ActivationFunctions.ReLU()
        self.expectedForwardPass = np.array([0, .32])
        self.expectedBackwardPass = np.array([-.156,.416])
        self.expectedGradient = np.array([[0,0],[-.416, .052]]).reshape(4)

    def testForwardPassNoBiasOrActivation(self):
        self.setUpNoBiasOrActivation()
        output = self.layerNoBiasOrActivation.forwardPass(self.input)
        for i, x in enumerate(self.expectedForwardPass):
            self.assertAlmostEqual(output[i], x, msg="Error in basic forward pass")
    
    def testForwardPassWithBiasNoActivation(self):
        self.setUpWithBiasNoActivation()
        output = self.layerWithBiasNoActivation.forwardPass(self.input)
        for i, x in enumerate(self.expectedForwardPass):
            self.assertAlmostEqual(output[i],x, msg="Bias values not added properly in forward pass")

    def testForwardPassNoBiasReLU(self):
        self.setUpNoBiasReLU()
        output = self.layerNoBiasReLU.forwardPass(self.input)
        for i, x in enumerate(self.expectedForwardPass):
            self.assertAlmostEqual(output[i],x, msg="ReLU activation failure in forward pass")
    
    def testBackwardPassNoBiasOrActivation(self):
        self.setUpNoBiasOrActivation()
        self.layerNoBiasOrActivation.forwardPass(self.input) #forward pass to properly set internal values
        output = self.layerNoBiasOrActivation.backwardPass(self.priorGradient)
        #check backward pass output
        for i, x in enumerate(self.expectedBackwardPass):
            self.assertAlmostEqual(output[i], x, 
            msg="Backward pass with no bias or activation fails to return proper gradient")
        #check weights gradient
        grad = self.layerNoBiasOrActivation.weights.grad.reshape(4)
        for i, x in enumerate(self.expectedGradient):
            self.assertAlmostEqual(grad[i], x, 
            msg="Backward pass with no bias or activation improperly updates weights gradient")
       
    def testBackwardPassWithBiasNoActivation(self):
        self.setUpWithBiasNoActivation()
        self.layerWithBiasNoActivation.forwardPass(self.input)
        output = self.layerWithBiasNoActivation.backwardPass(self.priorGradient)
        #same expected gradient, check gradient output
        for i, x in enumerate(self.expectedBackwardPass):
            self.assertAlmostEqual(output[i], x, 
            msg="Backwards pass with bias improperly passes back gradient")
        #weights gradient should be the same, check
        grad = self.layerWithBiasNoActivation.weights.grad.reshape(4)
        for i, x in enumerate(self.expectedGradient):
            self.assertAlmostEqual(grad[i], x, msg="Backwards pass with bias improperly sets weight gradient")
        #check bias gradient
        biasGrad = self.layerWithBiasNoActivation.bias.grad
        for i, x in enumerate(self.priorGradient):
            self.assertAlmostEqual(biasGrad[i], x, msg="Backwards pass with bias improperly sets bias grad")

    def testBackwardPassNoBiasReLU(self):
        self.setUpNoBiasReLU()
        #input = np.array([-.8,.1])
        self.layerNoBiasReLU.forwardPass(self.input)
        output = self.layerNoBiasReLU.backwardPass(self.priorGradient)
        #expect = np.array([-.156,.416])
        for i, x in enumerate(self.expectedBackwardPass):
            self.assertAlmostEqual(output[i], x, msg="backwards pass with relu improperly passes back gradient")
        #expectGrad = np.array([[0,0],[-.416, .052]]).reshape(4)
        grad = self.layerNoBiasReLU.weights.grad.reshape(4)
        for i, x in enumerate(self.expectedGradient):
            self.assertAlmostEqual(grad[i], x, msg="Backwards pass with relu improperly sets weight gradient")

    def testLayerInit(self):
        """
        Test different ways of initializing the network and ensure they properly set variables
        """
        #check layer of just input/output size
        layer = Layer(5,2)
        self.assertIsNone(layer.bias, "Shouldn't create bias unless it is specified")
        self.assertIsNone(layer.activationFunction, "Shouldn't create activation function without it being specified")
        #check layer with bias
        layer = Layer(5,2, bias = True)
        bias = layer.bias
        self.assertEqual(2, len(bias.bias))
        #make sure bias is intialized to 0
        for x in bias.bias:
            self.assertEqual(x, 0)
        self.assertIsNone(layer.activationFunction,
        msg="Shouldn't create activation function without it being specified")
        #check layer with activation function
        layer = Layer(5,2,activationFunction='ReLU')
        expect = ReLU()
        self.assertEqual(type(expect), type(layer.activationFunction), 
        msg="Should create ReLU activation function when activationFunction = ReLU specified")
        #check sigmoid
        layer = Layer(5,2, activationFunction='Sigmoid')
        expect = Sigmoid()
        self.assertEqual(type(expect), type(layer.activationFunction), 
        msg="Should create ReLU activation function when activationFunction = Sigmoid specified")
        #check leakyrelu
        layer = Layer(5,2, activationFunction='LeakyReLU')
        expect = LeakyReLU()
        self.assertEqual(type(expect), type(layer.activationFunction), 
        msg="Should create ReLU activation function when activationFunction = LeakyReLU specified")
        

class WeightsTests(unittest.TestCase):
    """
    Unit tests for Weights class. May not be needed if Layer tests are
    sufficient
    """
    def setUp(self):
        self.weightsTwoByTwo = Weights(2,2)
        self.weightsTwoByTwo._weights = np.array([[.1,.5],[-.3,.8]])
        self.input2 = np.array([.2,.4])
        self.prior2 = np.array([.44,.52])

    def testForwardPass(self):
        output = self.weightsTwoByTwo.forwardPass(self.input2)
        expect = np.array([.22,.26])
        for i, x in enumerate(expect):
            self.assertAlmostEqual(output[i],x)

    def testBackwardPassForProperOutputGradient(self):
        #assumes forwardPass works!
        #set gradient based on forward pass!
        dummy = self.weightsTwoByTwo.forwardPass(self.input2)
        output = self.weightsTwoByTwo.backwardPass(self.prior2)
        expect = np.array([-.112,.636])
        for i, x in enumerate(expect):
            self.assertAlmostEqual(output[i], x)

    def testBackwardPassForProperWeightGradient(self):
        dummy = self.weightsTwoByTwo.forwardPass(self.input2)
        output = self.weightsTwoByTwo.backwardPass(self.prior2)
        expectGrad = np.array([[.088,.176],[.104,.208]])
        actualGrad = self.weightsTwoByTwo.grad
        expectGrad = expectGrad.reshape(4)
        actualGrad = actualGrad.reshape(4)
        for i, x, in enumerate(expectGrad):
            self.assertAlmostEqual(actualGrad[i], x)

    @unittest.skip("Unwritten Test")
    def testUpdateGadient(self):
        pass