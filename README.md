# deformation-detection

# Files and what they are
## train_network.py:
this is the file that creates the model, trains and validates it, then saves the model along with a the accuracy/loss graph.
There is a function for data augmentation, but it is not called and it does not work.

## demonstrate_model_2.py:
This file is meant to demonstrate the model to Collingswood. It makes predictions about images and displays them in real time, but does not save the results.

## deploy_model_serverside_v0-7.py
This is a prototype of what we are going to put on our server. Right now it reads and writes to csv files, but when we get someone who specializes in web development/database management we will update it to communicate with the database on our server.

## CSV Files
The CSV files just contain coordinates and image names so that the network can know which files to look at.
