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
    def predictRNN(self, model, frame):
        
        y_hat = model.predict_classes(x=frame, verbose=1)
        print(y_hat.shape)
        pd.DataFrame(y_hat).to_csv('predicted.txt')
        exit()
        
        # del model
            
        return y_hat
    
    
    #------------------
    # Taining / Testing loop. All timesteps, whole image
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
    
    
    #------------------
    # Taining / Testing loop with "pixel strings"
    # Grabs batch_length pixel strings at a time and trains on them
    # Loop over tensor[i]'s xy-coordinates and
    # extract pixel strings from tensor
    # Feed into RNN for each xy
    # Batch Size: 1
    #-----------------
    def train_test_loop_pixel_string(self, model, tensor, importer, x, y, split, running_total, num_loops, restart, method, epochs):
        
        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                    
                print("x value: {}".format(x))
                print("y value: {}".format(y))
                                
                # pixel "string" & scale
                print("{}ing on coordinates: ({}, {})".format(method, x,y))
                pixel_string = importer.extractPixelStrings(tensor, x, y)

                x_train = np.array(pixel_string[:-1])
                y_train = np.array(pixel_string[-1])
                                        
                # One-hot everyone
                encoded_y = self.encode(y_train)
                # encoded_y = y_train
                    
                # Reshape training sequences for NN fit
                x_train = x_train.reshape(x_train.shape[0],1,1)
                x_train = np.swapaxes(x_train, 0, 1)
                encoded_y = np.array(encoded_y).reshape(1, len(encoded_y))
                    
                if method == 'Train':
                    trained = self.trainRNN(model, epochs, x_train, encoded_y)
                # else:
                #     tested = self.testRNN(model, epochs, tmp_x_batch, tmp_y_batch)

        return trained.history
    
    
    #------------------
    # Taining / Testing loop. OLD, one timestep at a time, per frame
    # Batch Size: tensor.shape[0]
    #-----------------
    def train_test_loop_old(self, model, tensor, method):
        
        
        for i in range(tensor.shape[2] - 2):        
                
            print("i value: {}".format(i))
            
            x_train = tensor[:, :, i]
            y_train = tensor[:, :, i+1]
                
            x_train = x_train.reshape(tensor.shape[0],tensor.shape[1],1)
            y_train = x_train.reshape(tensor.shape[0],tensor.shape[1],1)
            epochs = 1
                
            if method == 'Train':
                trained = self.trainRNN(model, epochs, x_train, y_train)
            # else:
            #     tested = self.testRNN(model, epochs, tmp_x_batch, tmp_y_batch)

        return True
    
    
    
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