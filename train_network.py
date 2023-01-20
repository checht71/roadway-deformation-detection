import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMG_HEIGHT = 1088
IMG_WIDTH = 1920
batch_size = 3
epochs = 4
DATADIR = '/home/christian/Desktop/Rowan CREATEs/Crack CNN/Data/Cracks'
model_output_string = "crack_detection_"+str(epochs)+".h5"



def augment_data(x, y):
    image = tf.image.stateless_random_brightness(x, max_delta=0.05)
    image = tf.image.stateless_random_flip_left_right
    image = tf.image.stateless_random_saturation
    return image, y
    

def plot_training_data():
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


model = keras.Sequential(
    [
        layers.Input((IMG_WIDTH, IMG_HEIGHT, 1)),
        layers.Conv2D(16, 3, padding="same"),
        layers.Conv2D(32, 3, padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
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
    seed = 123,
    validation_split=0.1,
    subset="training",
)


#train_dataset = train_dataset.map(augment_data)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATADIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(IMG_WIDTH, IMG_HEIGHT),  # reshape if not in this size
    shuffle=True,
    seed = 123,
    validation_split=0.1,
    subset="validation",
)


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)


history = model.fit(train_dataset, validation_data=validation_dataset,
                    epochs=epochs, verbose=1)

model.save(model_output_string)
plot_training_data()

predictions = model.predict(validation_dataset)
for x in range(len(predictions)):
    score = tf.nn.softmax(predictions[x])
    print(
        "Image "+str(x)+" most likely belongs to {} with a {:.2f} percent confidence."
        .format(validation_dataset.class_names[np.argmax(score)], 100 * np.max(score))
    )
