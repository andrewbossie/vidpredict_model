# VidPredict
<br>

## Description
VidPredict combines an LSTM with individual video frames to predict out-of-network 
images by modelling the temporal change of pixel values over time. 

## Versioning
Assumes python >= 3

## Installation
    pip install -r requirements.txt

## Run

Load video into /video.

    python vidpredict.py

Prediction images will be generated in /images/predicted_data

    
args: 

    noimport - Do not import any video data

    import - Import video data from /video

    predict - predict based on trained weights

## TODO
- Add support for AWS s3
- Metrics
- Visualizations

