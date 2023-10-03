# Label Assistant
# Use this to help you label images

import tensorflow as tf
import numpy as np
import os
import shutil



#Parameters
IMG_HEIGHT = 1088
IMG_WIDTH = 1920
DATASET_SIZE=15
DATASET_COLOR_CHANNELS = 1
batch_size = 3
class_names = [0, 1]
predictions = []
output_array = []
add_csv_data = True
copy = False
data_path = '/home/christian/Pictures/July Data/Images/'
model_location = '/home/christian/Documents/Pothole_detection_RC_2_DS5.h5'
new_directory = "/home/christian/Pictures/July Data/Images/Potholes/"

#append predictions list
def load_and_predict_image(img_path, loaded_model):
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode="grayscale"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = loaded_model.predict(img_array)
    softmax = tf.nn.softmax(predictions[0])
    for value in softmax:
        print(value)
    #Bias to 70% to indicate pothole rather than argmax (50%)
    threshold = 0.7  # Set the desired threshold

    # Find the index of the maximum softmax value
    max_index = np.argmax(softmax)
    
    # Check if the maximum softmax value exceeds the threshold for the second class
    if max_index == 1 and softmax[max_index] >= threshold:
        score = class_names[max_index]
    else:
        score = class_names[0]  # Default to the first class
        
    return score


def move_image(image_path, new_directory):
    if not os.path.isfile(image_path):
        print("Error: Image file not found.")
        return

    if not os.path.isdir(new_directory):
        os.makedirs(new_directory)

    image_filename = os.path.basename(image_path)
    new_path = os.path.join(new_directory, image_filename)

    if copy == True:
        shutil.copyfile(image_path, new_path)
    else:
         shutil.move(image_path, new_path)
   
    
    print("Image moved successfully to", new_path)




#Load in model)
model = tf.keras.models.load_model(
   model_location, custom_objects=None, compile=True,
   options=None
)

#Create output layer
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

street = os.listdir(data_path)

"""
for street in streets:
    file_names = os.listdir(data_path+street)
    file_names.sort()
"""

for j in range(len(street)):
    image_path = f'{data_path}/{street[j]}'
    try:
        deformation = load_and_predict_image(image_path, model)
        if deformation == 1:
            move_image(image_path, new_directory)
    except:
        print("FUCK")
print("Task Completed")
