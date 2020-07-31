#!/usr/bin/env python

# Title: VidPredict
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, July 2020

from classes.image import Image
from classes.rnn import RNN
from classes.visualizer import Visualizer
# from bayes import Bayes

import os
import sys
sys.setrecursionlimit(1000000)
import gc
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import boto3

# Program starts here
def VidPredict(do_import=True):
    
#----------------------------------
# Video import and image extraction
#----------------------------------

    # import all files from S3
    

    # For files in in /video
    for filename in os.listdir('../video'):
        
        index = 0
        
        importer = Image()
        
        if(do_import == True):
            print("Importing video file...")
            
            # Grab individual frames from video (limit 0 = all frames)
            importer.extractFrames(100, '../video/' + filename) 
            print("Done.")   
        
    # ----------------------------------
    # Loop through images and preprocess
    # ----------------------------------

        print("Grabbing pixel matrices and generating 3-d pixel tensor...")

        # Build 3-d tensor representing multiple image matrices (2-d)
        sorted_keys = []
        imageTensor = []
        
        # Explicitely sort filenames. 
        # This is important to keep temporal aspect of data
        for filename in os.listdir('../images/raw_data'):
            sorted_keys.append(int(filename.split('.')[0]))
            
        sorted_keys.sort()

        # For image in 'raw_data'
        for filename in sorted_keys:

            # Extract pixel matrices
            tmpMatrix = importer.getPixelMatrix('../../images/raw_data/' + str(filename) + ".jpg")
            tmpMatrix = tmpMatrix[:, :, 0]
            print("Saving pixel matrix...")
            # np.savetxt('tmp.txt', tmpMatrix, fmt='%i', delimiter=',')
            print("tmpMatrix for {} Shape: {}".format(str(filename) + ".jpg", tmpMatrix.shape))
            imageTensor.append(tmpMatrix)
            
        # Reshape tensor to have shape (samples, timesteps, features) OR (640, timesteps, 480)
        print("Before formatting: {}".format(np.asarray(imageTensor).shape))
        imageTensor = np.asarray(imageTensor).T
        imageTensor = np.swapaxes(np.asarray(imageTensor),2,1)
        print("imageTensor Shape: {}".format(np.asarray(imageTensor).shape))
        print("imageTensor Memory Taken: {} Mb".format(imageTensor.nbytes / 1000000))
        print("Done.")

        # RNN construction
        print("Building RNN...")
        
        # Split 'samples' (y-axis in this case, imageTensor.shape[0])
        split_len = imageTensor.shape[0]
        
        # Split entire data set (66/33)
        split = int(np.ceil(split_len * 0.33))
        print("Split value: {}".format(split))

        # Input Dim: imageTensor.shape[0] : imageTensor.shape[0] x imageTensor.shape[2] (samples:features:timesteps)
        # Output Dim: (480 x 640 x 1)
        # Output value: descrete scalar value 0 <  < 255
        rnn = RNN()
        model = rnn.buildRNN(imageTensor.shape[2], imageTensor.shape[1])
        model.summary()
        
        print("Done.")
        print("Starting main training loop...")

        # x = 0
        # y = 0
        # num_loops = 0
        # running_total = 0
        # batch_length = 1000
        # restart = True
        epochs = 200
        
    #-----------------
    # Train
    #-----------------

        method = 'Train'
        
        # Split image tensor into train tensor
        train_tensor = imageTensor[:-split, :, :]
        train_start = time.perf_counter()
        
        # Training via new method
        training_average = rnn.train_test_loop_new(model, 
                                            importer, 
                                            train_tensor,
                                            method,
                                            epochs)
        
        print("Training Complete.")
        train_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(train_end - train_start))
        
        
    #-----------------
    # Test
    #-----------------

        method = 'Test'
        epochs = 1
        
        # Split image tensor into train tensor
        train_tensor = imageTensor[split+1:, :, :]
        train_start = time.perf_counter()
        
        # Training via new method
        training_average = rnn.train_test_loop_new(model, 
                                            importer, 
                                            train_tensor,
                                            method,
                                            epochs)
        
        print("Training Complete.")
        train_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(train_end - train_start))
        
    #-----------------
    # Testing Metrics (Maybe)
    #-----------------
        # AUC
        # F1
        # Confusion Matrix

    #-----------------
    # Testing Visualizations (Maybe)
    #-----------------

    #-----------------
    # Predict
    #-----------------
    
        # Grab test image
        pred_image_location = "../../images/test_images/test.jpg"
        
        # Convert to image tensor
        pred_image_tensor = importer.getPixelMatrix(pred_image_location)
        pred_image_tensor = pred_image_tensor[:, :, 0]
        
        # Reshape tensor to have shape (samples, timesteps, features) OR (640, timesteps, 480)
        print("Before formatting: {}".format(np.asarray(pred_image_tensor).shape))
        pred_image_tensor = np.asarray(pred_image_tensor).T
        # test_image_tensor = np.swapaxes(np.asarray(pred_image_tensor),1,0)
        print("imageTensor Shape: {}".format(np.asarray(pred_image_tensor).shape))
        print("imageTensor Memory Taken: {} Mb".format(pred_image_tensor.nbytes / 1000000))
        print("Done.")
                
        # Build prediction rnn
        # timesteps = 1
        # features = pred_image_tensor[2]
        predModel = rnn.buildPredRNN(pred_image_tensor.shape[1], 1)
        predModel.summary()
        
        pred_image_tensor = pred_image_tensor.reshape((pred_image_tensor.shape[0], 1, pred_image_tensor.shape[1]))
        
        # Predict
        prediction = rnn.predictRNN(predModel, pred_image_tensor)
        
        # Convert predicted tensor to image and save
        final_prediction = importer.arrayToImage(prediction.T, index)
        index = index + 1
        
        print("Finished. Check /images/predicted_data for predictions.")
        
# One-off prediction based on already trained weights
def OneTimePredict(do_import=True):
    
    rnn = RNN()
    importer = Image()

    # Grab test image
    pred_image_location = "../../images/test_images/test.jpg"
    
    # Convert to image tensor
    pred_image_tensor = importer.getPixelMatrix(pred_image_location)
    pred_image_tensor = pred_image_tensor[:, :, 0]
    
    # Reshape tensor to have shape (samples, timesteps, features) OR (640, timesteps, 480)
    print("Before formatting: {}".format(np.asarray(pred_image_tensor).shape))
    pred_image_tensor = np.asarray(pred_image_tensor).T
    # test_image_tensor = np.swapaxes(np.asarray(pred_image_tensor),1,0)
    print("imageTensor Shape: {}".format(np.asarray(pred_image_tensor).shape))
    print("imageTensor Memory Taken: {} Mb".format(pred_image_tensor.nbytes / 1000000))
    print("Done.")
            
    # Build prediction rnn
    # timesteps = 1
    # features = pred_image_tensor[2]
    predModel = rnn.buildPredRNN(pred_image_tensor.shape[1], 1)
    predModel.summary()
    
    pred_image_tensor = pred_image_tensor.reshape((pred_image_tensor.shape[0], 1, pred_image_tensor.shape[1]))
    
    # Predict
    prediction = rnn.predictRNN(predModel, pred_image_tensor)
    
    # Convert predicted tensor to image and save
    final_prediction = importer.arrayToImage(prediction.T)
    
    print("Finished. Check /images/predicted_data for predictions.")
        

if __name__== "__main__":
    
    # Default
    if len(sys.argv) < 2: 
        VidPredict(True)
        
    # Explicit import
    elif sys.argv[1] == 'import':
        VidPredict(True)
        
    # Explicit No Import
    elif sys.argv[1] == 'noimport':
        VidPredict(False)
        
    # One-off predict 
    elif sys.argv[1] == 'predict':
        OneTimePredict(False)
