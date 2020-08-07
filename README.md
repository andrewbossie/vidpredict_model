# VidPredict

## Description
VidPredict combines an LSTM with individual video frames to predict out-of-network 
images by modelling the temporal change of pixel values over time. 

## Installation

Clone repository into desired location.

    pip install -r requirements.txt

* Assumes python >= 3

## Setup

Load your own training videos into 

    /video

Upload test image to 

    /images/test_images

Prediction images will be generated in 

    /images/predicted_data

## Run

    python vidpredict.py

    
args: 

    noimport - Do not import any video data

    import - Import video data from /video

    predict - quick predict based on trained weights

## TODO
- Add support for AWS s3
- Metrics
- Visualizations

