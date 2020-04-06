# Title: imageRNN
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, January 2020

import pandas as pd
import numpy as np 
import tensorflow as tf 
from tensorflow import keras

tf.executing_eagerly()
# viterbi - most likely sequence of hidden states.
# mini-forward
# gene sequencing?
# 255 x 255 state transition matrix
# 255 x 255 emission matrix: p(x1 | x-1) * P(x-1) for all states
# y_hat = MAX(emission matrix)


# This is the main class for bayesian inference (todo)
class Bayes(object):
    
    def __init__(self):
        
        # hyperparameters for building RNN
        self.pars = pars
        self.buildRNN(pars)