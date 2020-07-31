# VidPredict



## Description
VidPredict combines an LSTM with individual video frames to predict out-of-network 
images by modelling the temporal change of pixel values over time. 


## Versioning
Assumes python>=3


## Installation
    pip install -r requirements.txt


## Run

Load video into /video. Set a limit value for the amount of images you'd like to extract from the video.

    python vidpredict.py

or
    
    python vidpredict.py noimport


If you want to import a video files in /video run:

    python vidpredict.py import

If you only want to predict based on trained weights:

    python vidpredict.py predict

