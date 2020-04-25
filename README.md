NumberNet
=========
This project was created to improve my understanding of how neural networks work by implementing the code to train one. The code was built with the intention of recognizing handwritten digits from the MNIST database, although other datasets could be used (MNIST is included in the repository for convenience).

There's nothing special about this code so if you want to train something its probably best to go check out [TensorFlow](https://github.com/tensorflow/tensorflow)

The repository here consists of a few things
 - /NumberNet (the package which does everything)
 - main.py (has a demonstration of a basic network structure and uses of functions to run the network)
 - importData.py (a module used for importing data from MNIST)
 - /MNIST - directory containing MNIST data (original found [here](http://yann.lecun.com/exdb/mnist/))

I'll walk through what main does here, and implementations of everything are better documented in the code.
**main()**
This function does the training for the network. To do this a few things have to be set up:
 - Layer Structure
	 -
	 Layers are set up as a List of Layer objects. Each of these is made up of a shape, an optional bias, and an optional activation function. The code doesn't have very good checking for layer sizes so make sure they align properly!
 - Learning Parameters
	 -
	 A Parameters object that stores your desired hyperparameters. It allows you to set the step size, regularization strength, decay value, and by default RMSProp and momentum are both True.
 - Data imports
   -
   Using the functions from importData and the data found in MNIST, data is loaded in for use in training. You could change this to load in any data you desire, although it must follow the same conventions for labels as MNIST.

After these are set up, the Network can be created and trained. The Network class takes in Parameters, layers, and the loss function.
The initial accuracy is then evaluated by running the data through the network just after it is initialized.
The network is then trained on the training data with the specified batch size and number of epochs. (Since the entirety of this code is written in Python with the exception of the use of NumPy, this takes a little while)
 The network prints out some statistics on accuracy and how much training is left to do as it runs.
 Finally, the initial and final accuracy values are printed and the network object is saved out via pickle.

**showData()**
This function plays with the various ways to visualize what the network has done (and can do).
It starts by loading in the network from the pickled file and the runs the following 3 functions in succession
  - displayData
    -
    This function shows a matplotlib figure that shows the accuracy and loss during training on two separate subplots
    ![](https://i.imgur.com/T9KCGyZ.png)
 - visualizeWeights
   -
   This function is a bit hard coded to work with the layers set up in main(), but it shows the values of the first layer of the network. In a single layer network, you can clearly see it change from random noise to outlines of different numbers as you let the network train for longer amounts of time
   ![Fuzzy images](https://i.imgur.com/ap0cYZs.png)
   ![Images after sufficient training](https://i.imgur.com/JT6udHe.png)
 - run
   - 
   This runs the network! Once again, its tailored to MNIST. You can watch different images cycle past and it will display the truthy label as well as the guess the network makes for the given image.