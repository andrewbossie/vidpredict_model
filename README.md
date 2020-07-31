# VidPredict
<br>
<br>

## Description
VidPredict combines an LSTM with individual video frames to predict out-of-network 
images by modelling the temporal change of pixel values over time. 
<br>

## Versioning
Assumes python>=3
<br>

## Installation
    pip install -r requirements.txt
<br>

## Run

Load video into /video. Set a limit value for the amount of images you'd like to extract from the video.

    python vidpredict.py

    
args: 

    noimport - Do not import any video data

    import - Import video data from /video

    predict - predict based on trained weights

