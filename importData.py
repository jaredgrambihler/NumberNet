"""
Module to import data from MNIST dataset.
Dependencies:
    python-mnist
"""

from mnist import MNIST

def importData(dir = './mnist'):
    """
    Returns data of MNIST dataset.
    Uses the python-mnist module to import this data.
    If the directory of your data is different than /mnist,
    this method call can be edited.
    """
    try:
        #creates mndata object from mnist data
        mndata = MNIST(dir)

        #loads train data
        trainImages, trainLabels = mndata.load_training()

        #loads test data
        testImages, testLabels = mndata.load_testing()

        #returns as 4 lists
        return trainImages, trainLabels, testImages, testLabels
    except FileNotFoundError:
        print('Need to get MNIST data or change directory.')
        return None
    return None