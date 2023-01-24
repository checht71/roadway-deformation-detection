# Importing multiple models under one file presents issues!!!!


import tensorflow as tf
import numpy as np
import os
import csv
from tensorflow import keras

#Parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1920
DATASET_SIZE=15
DATASET_COLOR_CHANNELS = 1
batch_size = 3
class_names = [0, 1]
crack_predictions = []
pothole_predictions = []
output_array = []
PCI_BIAS = 1

#these are the file paths that must be edited to read and write to databases
data_path = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/SmallSet'
coordinate_path = './coordinates.csv'
output_path = '.output.csv'

#These are the locations of the models, I have to see if I can compress them in some way.
crack_model_location = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/Hydra II/models/crack_detection_4.h5'
pothole_model_location = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/Hydra II/models/pothole_detection_4.h5'


def load_and_predict(img_path, loaded_model):
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    score = class_names[np.argmax(score)]
    return score

def gather_and_write(current_predictions, second_loop, loaded_model):
    for lines in csv_data:
        image_path = data_path + "/" + lines[0]
        result = load_and_predict(image_path, loaded_model)
        if second_loop == False:
            output_array.append(lines[0]+','+lines[1]+','+
                          lines[2])
        current_predictions.append(result)
        
        

def import_model(model_location):
    model = tf.keras.models.load_model(
       crack_model_location, custom_objects=None, compile=True,
       options=None
    )
    
    #Create output layer
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
    return model


#Load data
dir_list = os.listdir(data_path)
print(dir_list)
csv_file = open(coordinate_path, mode ='r')
csv_data = list(csv.reader(csv_file))
csv_out = open(output_path, mode='a')

crack_model = import_model(crack_model_location)
gather_and_write(crack_predictions, False, crack_model)
keras.backend.clear_session()
pothole_model = import_model(pothole_model_location)
gather_and_write(pothole_predictions, True, pothole_model)

for y in range(len(crack_predictions)):
    PCI = crack_predictions[y]+pothole_predictions[y] + PCI_BIAS
    csv_out.write(str(output_array[y])+','+
                  str(crack_predictions[y])+','+
                  str(pothole_predictions[y])+','
                  +str(PCI)+'\n')

csv_file.close()
csv_out.close()
print("Task Completed")
