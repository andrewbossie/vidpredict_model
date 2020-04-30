# Title: imageRNN
# Author: Andrew Bossie
# Copyright (C) Andrew Bossie - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Andrew Bossie <andrewbossie06@yahoo.com>, January 2020

import numpy as np 
import pandas as pd 
import cv2
import PIL.Image
import os

#  The main class for image importing and data extaction
class Image(object):
    
    def __init__(self, filename):
        self.filename = filename
        
        # Dimensions
        self.x = 480
        self.y = 640
    
    # From a video file, extract the individual frames
    # Save to images/raw_data
    # Returns True
    def extractFrames(self, limit):
        
        video = cv2.VideoCapture(self.filename)
        success,image = video.read()
        
        index = 0
        success = True
        while success:
                
            cv2.imwrite("../images/raw_data/"+str(index)+".jpg", image)
            success,image = video.read()
            index += 1
            if limit > 0:
                if index == limit:
                    break
                
        
        return True
    
    # If x dimension is smaller than predetermined dimensions,
    # pad with zeros
    def padX(self, image):
        return True
    
    # If y dimension is smaller than predetermined dimensions,
    # pad with zeros
    def padY(self, image):
        return True
    
    # If x dimension is larger than predetermined dimensions,
    # crop to appropriate dimensions
    def cropX(self, image):
        return True
    
    # If y dimension is larger than predetermined dimensions,
    # crop to appropriate dimensions
    def cropY(self, image):
        return True
            
    # From a given still image, extract the corresponding pixel matrix
    # params: individual image name
    # Returns ndarray
    def getPixelMatrix(self, imageName):
        
        im = PIL.Image.open(imageName)
        
        # Resize to common dimensions
        im = im.resize((self.y, self.x))
            
        arr = np.array(im)
        
        #-----------------
        # Delete files in raw_data
        #-----------------
        os.remove(imageName)
        
        return arr
        
        
    # Take in an array of pixel arrays (tensor), extract 'strings of pixel values...
    # where a pixel 'ties' top the pixel with the same coordinate..
    # in the following frame (pixel array).
    # NOTE: the string of pixels is actually a 1-D array.
    # Returns ndarray
    def extractPixelStrings(self, tensor, x, y):
      
        # For each matrix, grab the value at coordinate and append to array
        pixelString = []
        count = 0
        for i in range(tensor.shape[2]):
            pixelString.append(tensor[x][y][i])
            count = count + 1
                
        pixelString = np.array(pixelString)
        return pixelString
    
        
    def arrayToImage(self, pixelArray):
        self.pixelArray = pixelArray
    
    