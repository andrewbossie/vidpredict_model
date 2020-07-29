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
        
        if(len(physical_devices) < 1):
            print("No GPU found... Exiting.")
            exit()
        
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
    
    # Train Function
    # If trained model is available, load weights
    def trainRNN(self, model, epochs, train_x, train_y):
        
        save_callback = ModelCheckpoint('../saved_models/trained.hdf5', 
                                                                  monitor='loss', 
                                                                  verbose=1, 
                                                                  save_best_only=False, 
                                                                  mode='auto', 
                                                                  save_freq=2)
        
        model.compile(loss='mse', optimizer='adam', metrics=['acc'])
        trained_rnn = model.fit(train_x, train_y, epochs=epochs,callbacks=[save_callback])
        # trained_rnn = model.fit(train_x, train_y, epochs=epochs)
            
        return trained_rnn
    
    # Test Function (Probably wont use)
    def testRNN(self, model, epochs, timesteps, train_x, train_y):
        
        tested_rnn = model.fit(train_x, train_y)
            
        return tested_rnn
    
    # Predict Function
    def predictRNN(self, model, pixel_value):
        
        y_hat = model.predict(255)
        print(y_hat)
        exit()
            
        return y_hat
    
    #------------------
    # Taining / Testing loop. Recursive
    # Grabs batch_length pixel strings at a time and trains on them
    # Loop over tensor[i]'s xy-coordinates and
    # extract pixel strings from tensor
    # Feed into RNN for each xy
    # Batch Size: 10000
    #-----------------
    def train_test_loop(self, model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart, method):
        
        print(tensor.shape)
        exit()
        
        # If we have iterated across x-axis (tensor.shape[0] - 1)
        if x == tensor.shape[0] - 2:
            final_average = (running_total / num_loops) * 100
            return final_average
        
        tmp_x_batch = []
        tmp_y_batch = []

        for i in range(y, y + batch_length):
            
            # If we have iterated across the y-axis - split
            if i == tensor.shape[1] - 2:
                x = x + 1
                # reset index
                i &= 0
                y = 0
                restart = True
                break
            
            else:
                y = i
                
            print("i value: {}".format(i))
            print("y value: {}".format(y))
            print("x value: {}".format(x))
                            
            # pixel "string" & scale
            print("{}ing on coordinates: ({}, {})".format(method, x,y))
            pixel_string = importer.extractPixelStrings(tensor, x, y)
            
            # Find length of pixel string for pair pixels
            modulo = len(pixel_string) % 2
            max_len = len(pixel_string) - modulo
            max_len = max_len - 1
            
            for k in range(max_len):
            
                x_train = pixel_string[k]
                y_train = pixel_string[k+1]
                print("K value: {}".format(k))
                print("K+1 value: {}".format(k+1))
                print("x_train value: {}".format(x_train))
                print("y_train value: {}".format(y_train))
                                    
                # One-hot everyone
                encoded_y = self.encode(y_train)
                
                x_train = x_train.reshape(1,1,1)
                encoded_y = np.array(encoded_y).reshape(-1 ,len(encoded_y))
                print("Expected output: {}".format(y_train))
                
                # tmp_x_batch.append(x_train)
                # tmp_y_batch.append(encoded_y)
                    
                # Train / test and loop back
                epochs = 1
                # tmp_x_batch = np.array(tmp_x_batch)
                # tmp_y_batch = np.array(tmp_y_batch)
                
                if method == 'Train':
                    trained = self.trainRNN(model, epochs, x_train, encoded_y)
                else:
                    tested = self.testRNN(model, epochs, tmp_x_batch, tmp_y_batch)
                    
                k = k + 2
                
        del tmp_x_batch
        del tmp_y_batch
        num_loops = num_loops + 1
        running_total = running_total + trained.history['acc'][0]
        return self.train_test_loop(model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart, method)
    
    
    #------------------
    # Prediction.
    # Given an input image,
    # generate the next image
    # using trained model.
    #-----------------
    def predict_loop(self, model, x, y, test_image_tensor):
        
        predicted_tensor = []

        for i in range(y):
            for j in range(x):
                predicted_tensor[j][i] = self.predictRNN(model, test_image_tensor[j][i])
        
        return predicted_tensor
            
    #------------------------------------------------
    # Encode y value into one hot (np.array(255))
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