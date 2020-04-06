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

#  The main class for image importing and data extaction
class Image(object):
    
    def __init__(self, filename):
        self.filename = filename
    
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
            
    # From a given still image, extract the corresponding pixel matrix
    # params: individual image name
    # Returns ndarray
    def getPixelMatrix(self, imageName):
        
        im = PIL.Image.open(imageName)
        arr = np.array(im)
        
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
    
    