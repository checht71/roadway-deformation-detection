#original name: deploy_model_serverside.py
import tensorflow as tf
import numpy as np
import os

#Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
DATASET_SIZE=15
DATASET_COLOR_CHANNELS = 1
batch_size = 3
class_names = [0, 1]
crack_predictions = []
pothole_predictions = []
output_array = []
add_csv_data = True

#these are the file paths that must be edited to read and write to databases
data_path = './Images/'
#These are the locations of the models, I have to see if I can compress them in some way.
crack_model_location = './Resnet50-P_last_20.h5'
def_type = 'Potholes'


#append predictions list
def load_and_predict_image(img_path, loaded_model):
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode="rgb"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    score = class_names[np.argmax(score)]
    return score


#Load in model)
model = tf.keras.models.load_model(
   crack_model_location, custom_objects=None, compile=True,
   options=None
)

#Create output layer
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

streets = os.listdir(data_path)
def_count_array = []

for street in streets:
    file_names = os.listdir(data_path+street)
    def_count = 0
        
    with open(f'./predictions/{street}_{def_type}_output.csv', "w") as g:
        for j in range(len(file_names)):
            image_path = f'{data_path}{street}/{file_names[j]}'
            crack = load_and_predict_image(image_path, model)
            PCI = float(crack)
            if crack == 1:
                def_count+=1

            g.write(f'{file_names[j]},{str(PCI)}\n')
    def_count_array.append(def_count)
            
for x in range(len(streets)):
    print(f"{def_type} on {streets[x]}: {def_count_array[x]}")
        
print("Task Completed")
