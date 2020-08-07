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

import os
import sys
sys.setrecursionlimit(1000000)
import gc
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
import boto3
# import seaborn as sb

# Program starts here
def VidPredict(do_import=True):
    
    rnn = RNN()
    normalizer = Normalizer()
    importer = Image()
    
#----------------------------------
# Video import and image extraction
#----------------------------------

    # import all files from S3
    

    index = 0
    
    # For files in in /video
    for filename in os.listdir('../video'):
        
        if(do_import == True):
            print("Importing video file...")
            
            # Grab individual frames from video (limit 0 = all frames)
            importer.extractFrames(10, '../video/' + filename) 
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
            if filename.split('.')[0] != 'README':
                sorted_keys.append(int(filename.split('.')[0]))
            
        sorted_keys.sort()

        # For image in 'raw_data'
        for filename in sorted_keys:

            # Extract pixel matrices
            tmpMatrix = importer.getPixelMatrix('../images/raw_data/' + str(filename) + ".jpg")
            tmpMatrix = tmpMatrix[:, :, 0]
            print("Saving pixel matrix...")
            # np.savetxt('tmp.txt', tmpMatrix, fmt='%i', delimiter=',')
            print("tmpMatrix for {} Shape: {}".format(str(filename) + ".jpg", tmpMatrix.shape))
            imageTensor.append(tmpMatrix)
            
        # Reshape tensor to have shape (samples, timesteps, features) OR (640, timesteps, 480)
        print("Before formatting: {}".format(np.asarray(imageTensor).shape))
        imageTensor = np.asarray(imageTensor).T
        # imageTensor = np.swapaxes(np.asarray(imageTensor),2,1)
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
        model = rnn.buildRNN(imageTensor.shape[2], imageTensor.shape[1])
        model.summary()
        
        print("Done.")
        print("Starting main training loop...")

        x = 0
        y = 0
        num_loops = 0
        running_total = 0
        restart = True
        epochs = 1
        
    #-----------------
    # Train
    #-----------------

        method = 'Train'
        
        # Split image tensor into train tensor
        train_tensor = imageTensor[:-split, :, :]
        train_start = time.perf_counter()
        
        # Training via new method
        training_average = rnn.train_test_predict_loop_pixel_string(model,
                                                            train_tensor,
                                                            importer, 
                                                            x, 
                                                            y, 
                                                            split, 
                                                            running_total, 
                                                            num_loops,
                                                            restart, 
                                                            method,
                                                            epochs)
        
        print("Training Complete.")
        train_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(train_end - train_start))
        
        
    #-----------------
    # Test (Unused)
    #-----------------

        method = 'Test'
        epochs = 1
        
        # Split image tensor into train tensor
        test_tensor = imageTensor[split+1:, :, :]
        test_start = time.perf_counter()
        
        # Testing via new method
        testing_average = rnn.train_test_predict_loop_pixel_string(model,
                                                            test_tensor,
                                                            importer, 
                                                            x, 
                                                            y, 
                                                            split, 
                                                            running_total, 
                                                            num_loops,
                                                            restart, 
                                                            method,
                                                            epochs,
                                                            normalizer)
        
        
        print("Training Complete.")
        test_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(test_end - test_start))
        
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
        pred_image_location = "../images/test_images/test.jpg"
        
        # Convert to image tensor
        pred_image_tensor = importer.getPixelMatrix(pred_image_location)
        pred_image_tensor = pred_image_tensor[:, :, 0]
        
        # Reshape tensor to have shape (samples, timesteps, features) OR (640, timesteps, 480)
        print("Before formatting: {}".format(np.asarray(pred_image_tensor).shape))
        pred_image_tensor = np.asarray(pred_image_tensor).T
        # test_image_tensor = np.swapaxes(np.asarray(pred_image_tensor),1,0)
        print("Prediction imageTensor Shape: {}".format(np.asarray(pred_image_tensor).shape))
        print("Prediction imageTensor Memory Taken: {} Mb".format(pred_image_tensor.nbytes / 1000000))
        print("Done.")
        
        pred_image_tensor = pred_image_tensor.reshape((pred_image_tensor.shape[0], 1, pred_image_tensor.shape[1]))
        print(pred_image_tensor.shape)
        # pd.DataFrame(pred_image_tensor).to_csv('before_preditcion.txt', index=True)
        # exit()
        
        # Predict
        method = 'Predict'
        
        # Split image tensor into train tensor
        pred_image_tensor = pred_image_tensor[split+1:, :, :]
        predict_start = time.perf_counter()
        
        # Testing via new method
        testing_average = rnn.train_test_predict_loop_pixel_string(model,
                                                            pred_image_tensor,
                                                            importer, 
                                                            x, 
                                                            y, 
                                                            split, 
                                                            running_total, 
                                                            num_loops,
                                                            restart, 
                                                            method,
                                                            epochs,
                                                            normalizer)
        
        
        print("Training Complete.")
        test_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(test_end - test_start))
        
        # Dump prediction tensor to file
        pd.DataFrame(prediction).to_csv('../images/predicted_data/pred_tensor.txt', index=True)
        
        # Convert predicted tensor to image and save
        final_prediction = importer.arrayToImage(prediction.T, index)
        index = index + 1
        
        print("Finished. Check /images/predicted_data for predictions.")
        
        exit()
        
# One-off prediction based on already trained weights
def OneTimePredict(do_import=True):
    
    rnn = RNN()
    importer = Image()

    # Grab test image
    pred_image_location = "../images/test_images/test.jpg"
    
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
    
    # Dump prediction tensor to file
    prediction.to_csv('../images/predicted_data/pred_tensor.txt', index=False)
    exit()
    
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
