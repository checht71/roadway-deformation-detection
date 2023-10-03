import wandb
import os
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import tensorflow_hub as hub
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_HEIGHT = 1088
IMG_WIDTH = 1920
batch_size = 3
epochs = 3
deformation_type = 'Pothole'
DATADIR = '/media/christian/USB STICK/Rowan CREATEs/Datasets/Pothole Images/PDS6/'
model_output_folder = "/home/christian/Documents/"
model_output_string = f'{deformation_type}_detection_{str(epochs)}_DS6.h5'
image_output_string = deformation_type+"_detection_"+str(epochs)+".png"
plot_title = deformation_type+" Detection Training"
class_names = [0, 1]
save_model = True
text_logging = False

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="CREATES Summer 2023")


data_augmentation = tf.keras.Sequential([
    # layers.RandomFlip("horizontal_and_vertical"),
    # layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomContrast(factor=0.2)
])


model = keras.Sequential(
    [
        layers.Input((IMG_WIDTH, IMG_HEIGHT, 1)),
        #data_augmentation,
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Dense(24),
        layers.Flatten(),
        layers.Dense(10),
        layers.Dense(2)
    ]
)


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(IMG_WIDTH, IMG_HEIGHT),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)


validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(IMG_WIDTH, IMG_HEIGHT),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    metrics=["accuracy"],
)


history = model.fit(train_dataset, validation_data=validation_dataset,
                    epochs=epochs, verbose=1,
                    callbacks=[
  WandbMetricsLogger(log_freq=5),
  WandbModelCheckpoint("models")])


if save_model == True:
    model.save(model_output_folder + model_output_string)
    print("Model Saved")
    print(model_output_string)

if text_logging == True:
    with open(f'{model_output_folder}/model_data.txt', 'a') as mout:
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
        mout.write(model_output_string)
        mout.write(f'\nTesting Accuracy: {accuracy}')
        mout.write(f'\nTesting loss: {loss}')
        mout.write(f'\nValidation Accuracy: {val_accuracy}')
        mout.write(f'\nValidation Loss: {val_loss}\n')
    

wandb.finish()
