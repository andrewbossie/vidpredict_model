# Title: imageRNN
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, January 2020

from image import Image
from rnn import RNN
from visualizer import Visualizer
# from bayes import Bayes

import os
import sys
sys.setrecursionlimit(1000000)
import gc
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Program starts here
def main(do_import=True):
    
#----------------------------------
# Video import and image extraction
#----------------------------------

    importer = Image('../video/CEP534.mpg')
    
    if(do_import == True):
        print("Importing video file...")
        
        # Grab individual frames from video (limit 0 = all frames)
        importer.extractFrames(limit=10) 
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
        tmpMatrix = importer.getPixelMatrix('../images/raw_data/' + str(filename) + ".jpg")
        tmpMatrix = tmpMatrix[:, :, 0]
        print("Saving pixel matrix...")
        np.savetxt('tmp.txt', tmpMatrix, fmt='%i', delimiter=',')
        print("tmpMatrix for {} Shape: {}".format(str(filename) + ".jpg", tmpMatrix.shape))
        imageTensor.append(tmpMatrix)
        
    # Dump 3-d Image Tensor into a csv
    print("Before formatting: {}".format(np.asarray(imageTensor).shape))
    
    imageTensor = np.asarray(imageTensor).T
    imageTensor = np.swapaxes(np.asarray(imageTensor),0,1)
    print("imageTensor Shape: {}".format(np.asarray(imageTensor).shape))
    print("imageTensor Memory Taken: {} Mb".format(imageTensor.nbytes / 1000000))
    print("Done.")

    # RNN construction
    print("Building RNN...")
    
    string_len = imageTensor.shape[2]
    
    # Split entire data set (66/33)
    split = int(np.ceil(string_len - (string_len * 0.33)))
    
    # Now lets split the training data into managable chunks (split / 10)

    # Input Dim: 1 x len(pixel string) : imageTensor.shape[2] x 1 (samples:timesteps:features)
    # Output Dim: 255
    # Output: descrete scalar value 0 < x < 255
    rnn = RNN()
    model = rnn.buildRNN(split-1, 1, 1)
    model.summary()
    
    print("Done.")

    print("Starting main training loop...")

    x = 0
    y = 0
    num_loops = 0
    running_total = 0
    batch_length = 1000
    
    # Extract pixel strings and train
    start = time.perf_counter()
    rnn.train_loop(model, importer, imageTensor, x, y, split, running_total, num_loops, batch_length, True)
    # print("Average Score: {}".format(final_average))
    print("Training Complete.")
    end = time.perf_counter()
    print("Model trained in {:.2f} seconds.".format(end - start))
    exit()
    
#-----------------
# Test
#-----------------
    # _test_loop(rnn, model, importer, imageTensor, x, y, split, running_total, num_loops, batch_length, True)
    
#-----------------
# Metrics
#-----------------

#-----------------
# Visualizations
#-----------------

# #------------------
# # Taining loop. Recursive
# # Grabs batch_length pixel strings at a time and trains on them
# # Loop over tensor[i]'s xy-coordinates and
# # extract pixel strings from tensor
# # Feed into RNN for each xy
# # Batch Size: 10000
# #-----------------
# def _train_loop(rnn, model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart):
    
#     # restart = True
#     if restart == True:
        
#         # print("x value: {}".format(x))
        
#         # If we have iterated across x-axis (tensor.shape[0] - 1)
#         # if x == 5:
#         if x == tensor.shape[0] - 1:
#             final_average = (running_total / num_loops) * 100
#             print("Final Average: {:.2f}".format(final_average))
#             return
        
#         tmp_x_batch = []
#         tmp_y_batch = []

#         for i in range(y, y + batch_length):
            
#             # If we have iterated across the y-axis
#             if i == tensor.shape[1]:
#                 x = x + 1
#                 # reset index
#                 i &= 0
#                 y = 0
#                 restart = True
#                 break
            
#             else:
#                 y = i
                
#             # print("i value: {}".format(i))
#             # print("y value: {}".format(y))
#             # print("x value: {}".format(x))
                            
#             # pixel "string" & scale
#             # print("Training on coordinates: ({}, {})".format(x,y))
#             pixel_string = importer.extractPixelStrings(tensor, x, y)
            
#             x_train = pixel_string[:split-1]
#             y_train = pixel_string[split]
#             x_test = pixel_string[:split+1]
#             y_test = pixel_string[split+2:]
                
#             timesteps = len(x_train)
            
#             # One-hot everyone
#             encoded_y = encode(y_train)
            
#             x_train = x_train.reshape(len(x_train),1)
#             encoded_y = np.array(encoded_y).reshape(len(encoded_y))
#             print("Expected output: {}".format(y_train))
            
#             tmp_x_batch.append(x_train)
#             tmp_y_batch.append(encoded_y)
                
#         # Train / test and loop back
#         epochs = 1
#         tmp_x_batch = np.array(tmp_x_batch)
#         tmp_y_batch = np.array(tmp_y_batch)
        
#         trained = rnn.trainRNN(model, epochs, timesteps, tmp_x_batch, tmp_y_batch)
#         print("Expected output: {}".format(y_train))
#         del tmp_x_batch
#         del tmp_y_batch
#         num_loops = num_loops + 1
#         running_total = np.mean(trained.history['acc'])
#         _train_loop(rnn, model, importer, tensor, x, y, split, running_total, num_loops, batch_length, restart)

# #------------------------------------------------
# # Encode y_hat value into one hot (np.array(255))
# #------------------------------------------------
# def encode(number):
#     encoded_array = []
    
#     for i in range(257):
#         if i != number:
#             encoded_array.append(0)
#         else:
#             encoded_array.append(1)
            
#     return encoded_array
            
            
            
if __name__== "__main__":
    if sys.argv[1] == 'import':
        main(True)
    elif sys.argv[1] == 'noimport':
        main(False)
