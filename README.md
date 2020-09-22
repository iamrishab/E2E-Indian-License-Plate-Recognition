# License Plate Recognition

## Overview
Currently this code includes:

**Detection models:**

1. SSD-Mobilenet

**Text Recognition:**

1. Tesseract

## Install APT packages
$ sudo apt-get install tesseract-ocr

## Run the code in virtual environment (Recommended)
$ pip3 install virtualenv

$ virtualenv lprv1_env

$ source lprv1_env/bin/activate

$ pip3 install -r requirements.txt

$ git clone https://rishab-pal-onebcg@bitbucket.org/onebcg/licenseplaterecognition.git

$ cd licenseplaterecognition

$ git checkout dev/Phase1-Rishab

## Download Resource
Download and place the content from the [link](https://drive.google.com/drive/folders/1gdJDciujoEcVbUSb640rzBuPDy8UqNWr?usp=sharing) and place it inside `LPRv1/model` subfolder.

## To test via API
$ python api.py

**Swagger API**

http://0.0.0.0:6500/

## To test on video

$ python test.py


**Note: This code is compatible with Python >= 3.6.0.**
Although it might work with other Python versions with some minor changes