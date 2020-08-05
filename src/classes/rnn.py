# Title: VidPredict
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, July 2020

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
        
        # Are we using GPU? - REQUIRED! -
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs:", len(physical_devices))
        
        if(len(physical_devices) < 1):
            print("No GPU found... Exiting.")
            exit()
            
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        
        return None
    
    # Build new neural network for training and testing
    def buildRNN(self, features, timesteps):
        
        rnn = keras.Sequential()
        # rnn.add(LSTM(256, return_sequences=True, input_shape=(timesteps-2, features)))
        rnn.add(LSTM(256, return_sequences=True, input_shape=(None, 1)))
        rnn.add(Dropout(0.2))
        rnn.add(LSTM(128, return_sequences=True))
        rnn.add(Dropout(0.2))
        rnn.add(LSTM(56))
        rnn.add(Dropout(0.2))
        rnn.add(Dense(257, activation='softmax'))
        
        # If model already exists
        if path.exists('../../saved_models/trained.hdf5'):
            print('Existing model found, loading...')

            # Restore the weights
            rnn.load_weights('../../saved_models/trained.hdf5')
            print('Done.')
            
        return rnn
    
    # Build new neural network for prediction
    # def buildPredRNN(self, features, timesteps):
        
    #     rnn = keras.Sequential()
    #     # rnn.add(LSTM(256, return_sequences=True, input_shape=(timesteps-2, features)))
    #     rnn.add(LSTM(256, return_sequences=True, input_shape=(timesteps, features)))
    #     rnn.add(Dropout(0.2))
    #     rnn.add(LSTM(128, return_sequences=True))
    #     rnn.add(Dropout(0.2))
    #     rnn.add(LSTM(56))
    #     rnn.add(Dropout(0.2))
    #     rnn.add(Dense(features, activation='softmax'))
        
    #     # If model already exists
    #     if path.exists('../../saved_models/trained.hdf5'):
    #         print('Existing model found, loading...')

    #         # Restore the weights
    #         rnn.load_weights('../../saved_models/trained.hdf5')
    #         print('Done.')
            
    #     return rnn
    
    # Train Function
    def trainRNN(self, model, epochs, train_x, train_y):
        
        save_callback = ModelCheckpoint('../saved_models/trained.hdf5', 
                                                                  monitor='loss', 
                                                                  verbose=1, 
                                                                  save_best_only=False, 
                                                                  mode='auto', 
                                                                  save_freq=2)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'categorical_crossentropy'])
        # model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        trained_rnn = model.fit(train_x, train_y, epochs=epochs,callbacks=[save_callback])
            
        return trained_rnn
    
    # Test Function (Probably wont use)
    def testRNN(self, model, epochs, timesteps, train_x, train_y):
        
        tested_rnn = model.fit(train_x, train_y)
            
        return tested_rnn
    
    # Predict Function
    def predictRNN(self, model, x):
        
        y_hat = model.predict_classes(x=x, verbose=1)
        
        # inverse one hot
        
        del model
        return y_hat
        
    
    #------------------
    # Taining / Testing loop with "pixel strings"
    # Grabs batch_length pixel strings at a time and trains on them
    # Loop over tensor[i]'s xy-coordinates and
    # extract pixel strings from tensor
    # Feed into RNN for each xy
    # Batch Size: 1
    #-----------------
    def train_test_predict_loop_pixel_string(self, model, tensor, importer, x, y, split, running_total, num_loops, restart, method, epochs):
        
        # If we are predicting, initialize array
        if method == 'Predict':
            predict_array = np.array()
        
        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                    
                print("x value: {}".format(x))
                print("y value: {}".format(y))
                                
                # pixel "string" & scale
                print("{}ing on coordinates: ({}, {})".format(method, x,y))
                pixel_string = importer.extractPixelStrings(tensor, x, y)

                x_batch = np.array(pixel_string[:-1])
                y_batch = np.array(pixel_string[-1])
                                        
                # One-hot labels
                encoded_y = self.encode(y_batch)
                # encoded_y = y_batch
                    
                # Reshape training sequences for LSTM fit
                x_batch = x_batch.reshape(x_batch.shape[0],1,1)
                x_batch = np.swapaxes(x_batch, 0, 1)
                encoded_y = np.array(encoded_y).reshape(1, len(encoded_y))
                    
                if method == 'Train':
                    trained = self.trainRNN(model, epochs, x_batch, encoded_y)
                elif method == 'Test':
                    tested = self.testRNN(model, epochs, x_batch, encoded_y)
                elif method == 'Predict':
                    predicted = self.predictRNN(model, tmp_x_batch)
                    predicted_array[x,y] = predicted
                else:
                    print("No Method Defined! Exiting...")
                    exit()

        if method == 'Predict':
            return trained.history, predicted_array
        else:
            return trained.history
    
    
    #------------------
    # Taining / Testing loop. All timesteps, whole image (DEPRECATED)
    # Batch Size: tensor.shape[0]
    #-----------------
    def train_test_loop_new(self, model, importer, tensor, method, epochs):
        
        x_train = tensor[:, :tensor.shape[1]-2, :]
        y_train = tensor[:, tensor.shape[1]-1, :]
            
        if method == 'Train':
            trained = self.trainRNN(model, epochs, x_train, y_train) 
        # else:
        #     tested = self.testRNN(model, epochs, tmp_x_batch, tmp_y_batch)
        
        del model
            
        return True
    
            
    #------------------------------------------------
    # Encode y value into one hot (np.array(255)) (DEPRECATED)
    #------------------------------------------------
    def encode(self, number):

        encoded_array = []
        
        for i in range(257):
            if i != number:
                encoded_array.append(0)
            else:
                encoded_array.append(1)
                
        return encoded_array
    
    #------------------------------------------------
    # Decode y_hat (nparray) value into int 0 < x < 255
    #------------------------------------------------
    def decode(self, y_hat):
        
        final_value = y_hat.index(1) - 1
                
        return final_value
            
    def destroyRNN(self):
        keras.backend.clear_session()