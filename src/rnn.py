# Title: imageRNN
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, January 2020

import os
import os.path
from os import path
import time
import pandas as pd
import numpy as np 
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

tf.executing_eagerly()

# This is the main class for recurrent neural network
class RNN(object):
    
    def __init__(self):
        
        # Are we using GPU?
        physical_devices = tf.config.list_physical_devices('GPU') 
        print("Num GPUs:", len(physical_devices))
        
        return None
    
    # Build new neural network
    def buildRNN(self, samples, timesteps, features):
        
        rnn = keras.Sequential()
        rnn.add(LSTM(256, return_sequences=True, input_shape=(samples, 1)))
        rnn.add(Dropout(0.2))
        rnn.add(LSTM(128))
        rnn.add(Dropout(0.2))
        rnn.add(Dense(257, activation='relu'))
        
        # If model already exists
        if path.exists('../saved_models/trained.hdf5'):
            print('Existing model found, loading...')

            # Restore the weights
            rnn.load_weights('../saved_models/trained.hdf5')
            print('Done.')
            
        return rnn
    
    def trainRNN(self, model, epochs, timesteps, train_x, train_y):
        
        save_callback = ModelCheckpoint('../saved_models/trained.hdf5', 
                                                                  monitor='loss', 
                                                                  verbose=1, 
                                                                  save_best_only=False, 
                                                                  mode='auto', 
                                                                  save_freq=1)
        
        model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        trained_rnn = model.fit(train_x, train_y, epochs=epochs,callbacks=[save_callback])
        # trained_rnn = model.fit(train_x, train_y, epochs=epochs)
            
        return trained_rnn
    
    #------------------
    # Taining loop. Recursive
    # Grabs batch_length pixel strings at a time and trains on them
    # Loop over tensor[i]'s xy-coordinates and
    # extract pixel strings from tensor
    # Feed into RNN for each xy
    # Batch Size: 10000
    #-----------------
    def train_loop(self, model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart):
        
        # if restart == True:
            
        # print("x value: {}".format(x))
        
        # If we have iterated across x-axis (tensor.shape[0] - 1)
        if x == 5:
        # if x == tensor.shape[0] - 1:
            final_average = (running_total / num_loops) * 100
            return final_average
        
        tmp_x_batch = []
        tmp_y_batch = []

        for i in range(y, y + batch_length):
            
            # If we have iterated across the y-axis
            if i == tensor.shape[1]:
                x = x + 1
                # reset index
                i &= 0
                y = 0
                restart = True
                break
            
            else:
                y = i
                
            # print("i value: {}".format(i))
            # print("y value: {}".format(y))
            # print("x value: {}".format(x))
                            
            # pixel "string" & scale
            # print("Training on coordinates: ({}, {})".format(x,y))
            pixel_string = importer.extractPixelStrings(tensor, x, y)
            
            x_train = pixel_string[:split-1]
            y_train = pixel_string[split]
            # x_test = pixel_string[:split+1]
            # y_test = pixel_string[split+2:]
                
            timesteps = len(x_train)
            
            # One-hot everyone
            encoded_y = self.encode(y_train)
            
            x_train = x_train.reshape(len(x_train),1)
            encoded_y = np.array(encoded_y).reshape(len(encoded_y))
            print("Expected output: {}".format(y_train))
            
            tmp_x_batch.append(x_train)
            tmp_y_batch.append(encoded_y)
                
        # Train / test and loop back
        epochs = 1
        tmp_x_batch = np.array(tmp_x_batch)
        tmp_y_batch = np.array(tmp_y_batch)
        
        trained = self.trainRNN(model, epochs, timesteps, tmp_x_batch, tmp_y_batch)
        print("Expected output: {}".format(y_train))
        del tmp_x_batch
        del tmp_y_batch
        num_loops = num_loops + 1
        running_total = running_total + trained.history['acc'][0]
        return self.train_loop(model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart)
            
    #------------------------------------------------
    # Encode y_hat value into one hot (np.array(255))
    #------------------------------------------------
    def encode(self, number):
        encoded_array = []
        
        for i in range(257):
            if i != number:
                encoded_array.append(0)
            else:
                encoded_array.append(1)
                
        return encoded_array
            
    def destroyRNN(self):
        keras.backend.clear_session()
        
    def predictFrame(self, x):
        self.x = x