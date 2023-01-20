import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import PIL
import csv

#Parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1920
DATASET_SIZE=15
DATASET_COLOR_CHANNELS = 1
batch_size = 3
model_location = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/Hydra II/models/pothole_detection_4.h5'
data_path = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/SmallSet'
coordinate_path = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/Hydra II/Demonstration/East Linden Ave Coordinates 1-63.csv'
class_names = [0, 1]
predictions = []
PLOT_SIZE = 9

def load_and_predict(img_path):
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




#Import model
loaded_model = tf.keras.models.load_model(
   model_location, custom_objects=None, compile=True,
   options=None
)

#Create output layer
probability_model = tf.keras.Sequential([loaded_model, 
                                         tf.keras.layers.Softmax()])

#Load data
dir_list = os.listdir(data_path)
csv_file = open('East Linden Ave Coordinates 1-63.csv', mode ='r')
csv_data = list(csv.reader(csv_file))

#GEO CSV contains filenames to analyze
for lines in csv_data:
    image_path = data_path + "/" + lines[0]
    print(image_path)
    
    result = load_and_predict(image_path)
    predictions.append(result)
    
    if result == 1:
        img_title = "Crack Detected"
    else:
        img_title = "No Crack Detected"
    
    plt.title(img_title)
    image_show = PIL.Image.open(image_path)
    plt.imshow(image_show)
    plt.figtext(0.5, 0.01, str(lines),wrap=True, 
                horizontalalignment='center',
                fontsize=12)
    plt.show()

csv_file.close()