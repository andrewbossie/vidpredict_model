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

    # For files in in /video
    for filename in os.listdir('../video'):
        importer = Image(filename)
        
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
            tmpMatrix = importer.getPixelMatrix('../images/raw_data/' + str(filename) + ".jpeg")
            tmpMatrix = tmpMatrix[:, :, 0]
            print("Saving pixel matrix...")
            # np.savetxt('tmp.txt', tmpMatrix, fmt='%i', delimiter=',')
            print("tmpMatrix for {} Shape: {}".format(str(filename) + ".jpeg", tmpMatrix.shape))
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
        
        split_len = imageTensor.shape[0]
        
        # Split entire data set (66/33)
        split = int(np.ceil(split_len * 0.33))
        print("Split value: {}".format(split))
        print(imageTensor.shape)
        exit()

        # Input Dim: 1 x len(pixel string) : imageTensor.shape[2] x 1 (samples:timesteps:features)
        # Output Dim: 255
        # Output: descrete scalar value 0 < x < 255
        rnn = RNN()
        model = rnn.buildRNN(imageTensor.shape[2]-1, 1, 1)
        model.summary()
        
        print("Done.")
        print("Starting main training loop...")

        x = 0
        y = 0
        num_loops = 0
        running_total = 0
        batch_length = 1000
        restart = True
        
    #-----------------
    # Test
    #-----------------

        method = 'Train'
        
        # Split image tensor into train tensor
        train_tensor = imageTensor[:-split][:][:]
        train_start = time.perf_counter()
        
        training_average = rnn.train_test_loop(model, 
                                            importer, 
                                            train_tensor, 
                                            x, 
                                            y, 
                                            split, 
                                            running_total, 
                                            num_loops, 
                                            batch_length, 
                                            restart, 
                                            method)
        
        print("Average Training Score: {:.2f}%".format(training_average))
        print("Training Complete.")
        train_end = time.perf_counter()
        print("Model trained in {:.2f} seconds.".format(train_end - train_start))
        
        
    #-----------------
    # Test
    #-----------------

        # method = 'Test'
        
        # # Split image tensor into test tensor
        # test_tensor = imageTensor[-split+1:][:][:]
        # test_start = time.perf_counter()
        
        # testing_average = rnn.train_test_loop(model, 
        #                                       importer, 
        #                                       test_tensor, 
        #                                       x, 
        #                                       y, 
        #                                       split, 
        #                                       running_total, 
        #                                       num_loops, 
        #                                       batch_length, 
        #                                       restart, 
        #                                       method)
        
        # print("Average Testing Score: {:.2f}%".format(testing_average))
        # print("Testing Complete.")
        # test_end = time.perf_counter()
        # print("Model tested in {:.2f} seconds.".format(test_end - test_start))
        
    #-----------------
    # Testing Metrics
    #-----------------
        # AUC
        # F1
        # Confusion Matrix

    #-----------------
    # Testing Visualizations
    #-----------------

    #-----------------
    # Predictions (TEST)
    #-----------------
        # prediction = rnn.predict_loop()
        

if __name__== "__main__":
    if sys.argv[1] == 'import':
        main(True)
    elif sys.argv[1] == 'noimport':
        main(False)
