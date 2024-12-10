#!/bin/bash

# install packages 
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install torchvision tensorboard

# untar the test and training data
tar zxf data.tar.gz

# run the pytorch model
python3 main.py ./ --epochs=1

# remove the data directory
rm -r data