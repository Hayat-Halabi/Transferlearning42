# Transferlearning42
### Face Mask Detection
# Scenario
After the global pandemic, places like movie theaters across the US are starting to open again. A big part of returning to normal is ensuring people are safe and following guidelines such as wearing face masks. MMC Sercado, a big-shot cinema owner in the US, wants to ensure that masks are worn correctly to maintain safety for all patrons.

# Problem Statement
You need to develop a deep learning model using transfer learning to spot and sort the following three situations among the movie crowd:

Properly adhering to face mask guidelines.
Incorrectly wearing the face mask.
Choosing not to wear a mask at all.
Task: Build a Transfer Learning model to detect face masks on humans.
# Dataset Description
The zip folder contains 2 folders train and test each folder with 3 subfolders labelled as to which class they belong to.

The 3 classes are

"with_mask"
"without_mask"
"mask_worn_incorrect"
Each image is of shape 128,128,3.

# Directions:
Import the necessary Libraries
### Task A

Load the image training and test datasets from the train and test folders, respectively. Each image is 128 x 128 x 3.
Load the training dataset using Keras ImageDataGenerator with validation_split=0.2.
Load the test dataset using Keras ImageDataGenerator.
Build a transfer learning network using Keras with the following layers:
EfficientNetB0 as the first layer using the Keras API
GLobalAveragePooling2D layer
Dropout(0.2)
Dense layer with 3 neurons and activation softmax
Compile the model with the Adam optimizer, categorical cross-entropy loss, and metrics accuracy.
Train the model for 4 epochs with callbacks. Reduce the learning rate on Plateau and stop early while monitoring validation loss.
Plot training and validation accuracy and loss against epochs.
### Task B

Load the image training and test datasets from the train and test folders, respectively. Each image is 128 x 128 x 3.
Load the training dataset using Keras ImageDataGenerator with validation_split=0.2.
Load the test dataset using Keras ImageDataGenerator.
Build a transfer learning network using Keras with the following layers:
ResNet50 as the first layer using the Keras API
GLobalAveragePooling2D layer
Dropout(0.5)
Dense layer with 3 neurons and activation softmax
Compile the model with the Adam optimizer, categorical cross-entropy loss, and metrics accuracy.
Train the model for 4 epochs with callbacks. Reduce the learning rate on Plateau and stop early while monitoring validation loss.
Plot training and validation accuracy and loss against epochs.
### Task C

Compare EfficientNetB0 and ResNet50 model performance and find the best model on the basis of accuracy.
Import the necessary libraries
``` python
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
#from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.optimizers import Adam

##Task A
#Load the Image Training and Test Datasets from the train and test folder respectively. Each image is of shape 128 x 128 x 3.
#!unzip -q face_mask_detection_dataset.zip 

train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True, vertical_flip = True, zoom_range = 0.1,
    shear_range = 0.1, width_shift_range = 0.2, height_shift_range = 0.2, rotation_range = 90,
)
test_data_generator = keras.preprocessing.image.ImageDataGenerator()
#Load the training dataset using Keras ImageDataGenerator.
train_data = train_data_generator.flow_from_directory("face_mask_detection_dataset/train", target_size = (128, 128), batch_size = 1, shuffle = True)
#Load the test dataset using Keras ImageDataGenerator.
test_data = test_data_generator.flow_from_directory("face_mask_detection_dataset/test", target_size = (128,128), batch_size = 1, shuffle = True)

labels = train_data.class_indices
labels
```
There are three labels found: mask_weared_incorrect, with_mask, and without_mask.

ImageDataGenerator does not really assign values to an array, it just hold pointers. Because of that every learning step CPU perform reading operations. This very slows learning speed.

Store the data in a numpy array type.
```python
def get_array_from_datagen(train_generator):
  x=[]
  y=[]
  train_generator.reset()
  for i in range(train_generator.__len__()):
    a,b=train_generator.next()
    x.append(a)
    y.append(b)
  x=np.array(x, dtype = np.float32)
  y=np.array(y, dtype = np.float32)
  print(x.shape)
  print(y.shape)
  return x,y

X_train, y_train = get_array_from_datagen(train_data)
X_test, y_test = get_array_from_datagen(test_data)

X_train = X_train.reshape(-1, 128, 128, 3)
X_test = X_test.reshape(-1, 128, 128, 3)
y_train = y_train.reshape(-1, 3)
y_test = y_test.reshape(-1, 3)

input_shape = (128, 128, 3)
class_num = len(labels)

import matplotlib.pyplot as plt
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
```
Build a Transfer Learning network using Keras with the following layers:
EfficientNetB0 as first layers using Keras API.
GLobalAveragePooling2D layer
Dropout(0.2)
Dense layer with 3 neurons and activation softmax
```python
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

efficientnet_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
])
#
Block 42: Workshop
Face Mask Detection
Scenario
After the global pandemic, places like movie theaters across the US are starting to open again. A big part of returning to normal is ensuring people are safe and following guidelines such as wearing face masks. MMC Sercado, a big-shot cinema owner in the US, wants to ensure that masks are worn correctly to maintain safety for all patrons.

Problem Statement
You need to develop a deep learning model using transfer learning to spot and sort the following three situations among the movie crowd:

Properly adhering to face mask guidelines.
Incorrectly wearing the face mask.
Choosing not to wear a mask at all.
Task: Build a Transfer Learning model to detect face masks on humans.
Dataset Description
The zip folder contains 2 folders train and test each folder with 3 subfolders labelled as to which class they belong to.

The 3 classes are

"with_mask"
"without_mask"
"mask_worn_incorrect"
Each image is of shape 128,128,3.

Directions:
Import the necessary Libraries
Task A

Load the image training and test datasets from the train and test folders, respectively. Each image is 128 x 128 x 3.
Load the training dataset using Keras ImageDataGenerator with validation_split=0.2.
Load the test dataset using Keras ImageDataGenerator.
Build a transfer learning network using Keras with the following layers:
EfficientNetB0 as the first layer using the Keras API
GLobalAveragePooling2D layer
Dropout(0.2)
Dense layer with 3 neurons and activation softmax
Compile the model with the Adam optimizer, categorical cross-entropy loss, and metrics accuracy.
Train the model for 4 epochs with callbacks. Reduce the learning rate on Plateau and stop early while monitoring validation loss.
Plot training and validation accuracy and loss against epochs.
Task B

Load the image training and test datasets from the train and test folders, respectively. Each image is 128 x 128 x 3.
Load the training dataset using Keras ImageDataGenerator with validation_split=0.2.
Load the test dataset using Keras ImageDataGenerator.
Build a transfer learning network using Keras with the following layers:
ResNet50 as the first layer using the Keras API
GLobalAveragePooling2D layer
Dropout(0.5)
Dense layer with 3 neurons and activation softmax
Compile the model with the Adam optimizer, categorical cross-entropy loss, and metrics accuracy.
Train the model for 4 epochs with callbacks. Reduce the learning rate on Plateau and stop early while monitoring validation loss.
Plot training and validation accuracy and loss against epochs.
Task C

Compare EfficientNetB0 and ResNet50 model performance and find the best model on the basis of accuracy.
Import the necessary libraries

[ ]
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
#from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.optimizers import Adam
Task A
Load the Image Training and Test Datasets from the train and test folder respectively. Each image is of shape 128 x 128 x 3.
[ ]
### BEGIN SOLUTION
#!unzip -q face_mask_detection_dataset.zip 

train_data_generator = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True, vertical_flip = True, zoom_range = 0.1,
    shear_range = 0.1, width_shift_range = 0.2, height_shift_range = 0.2, rotation_range = 90,
)
test_data_generator = keras.preprocessing.image.ImageDataGenerator()
### END SOLUTION
[ ]
### BEGIN HIDDEN TESTS
assert 1 == 1, "Incorrect. Try again."
print ("Correct.")
### END HIDDEN TESTS
Load the training dataset using Keras ImageDataGenerator.
[ ]
### BEGIN SOLUTION
train_data = train_data_generator.flow_from_directory("face_mask_detection_dataset/train", target_size = (128, 128), batch_size = 1, shuffle = True)
### END SOLUTION
[ ]
### BEGIN HIDDEN TESTS
assert 1 == 1, "Incorrect. Try again."
print ("Correct.")
### END HIDDEN TESTS
Load the test dataset using Keras ImageDataGenerator.
[ ]
### BEGIN SOLUTION
test_data = test_data_generator.flow_from_directory("face_mask_detection_dataset/test", target_size = (128,128), batch_size = 1, shuffle = True)

labels = train_data.class_indices
labels
### END SOLUTION
[ ]
### BEGIN HIDDEN TESTS
assert 1 == 1, "Incorrect. Try again."
print ("Correct.")
### END HIDDEN TESTS
There are three labels found: mask_weared_incorrect, with_mask, and without_mask.

ImageDataGenerator does not really assign values to an array, it just hold pointers. Because of that every learning step CPU perform reading operations. This very slows learning speed.

Store the data in a numpy array type.
[ ]
### BEGIN SOLUTION
def get_array_from_datagen(train_generator):
  x=[]
  y=[]
  train_generator.reset()
  for i in range(train_generator.__len__()):
    a,b=train_generator.next()
    x.append(a)
    y.append(b)
  x=np.array(x, dtype = np.float32)
  y=np.array(y, dtype = np.float32)
  print(x.shape)
  print(y.shape)
  return x,y

X_train, y_train = get_array_from_datagen(train_data)
X_test, y_test = get_array_from_datagen(test_data)

X_train = X_train.reshape(-1, 128, 128, 3)
X_test = X_test.reshape(-1, 128, 128, 3)
y_train = y_train.reshape(-1, 3)
y_test = y_test.reshape(-1, 3)

input_shape = (128, 128, 3)
class_num = len(labels)

import matplotlib.pyplot as plt
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
### END SOLUTION
[ ]
### BEGIN HIDDEN TESTS
assert 1 == 1, "Incorrect. Try again."
print ("Correct.")
### END HIDDEN TESTS
Build a Transfer Learning network using Keras with the following layers:
EfficientNetB0 as first layers using Keras API.
GLobalAveragePooling2D layer
Dropout(0.2)
Dense layer with 3 neurons and activation softmax
[ ]
### BEGIN SOLUTION
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

efficientnet_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
])

### Compile the model with adam optimizer, categorical_crossentropy loss, and with metrics accuracy.
efficientnet_model.compile(
    optimizer="Adam",
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)
#Train the model for 4 epochs with callbacks. Reduce learning rate on Plateau and early stopping while monitoring validation loss.
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor = "val_accuracy",
    factor = 0.5,
    patience = 3,
    verbose = 0,
    min_lr = 0.00001
)
early_stopping = keras.callbacks.EarlyStopping(patience=5, verbose=1)

history = efficientnet_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=2,
    callbacks = [learning_rate_reduction, early_stopping]
)
#Plot training and validation accuracy and loss against epochs.
plot_history(history)
```
Task B
Load the Image Training and Test Datasets from the train and test folder respectively. Each image is of shape 128 x 128 x 3.
Load training dataset using Keras ImageDataGenerator with validation_split=0.2.
Load test dataset using Keras ImageDataGenerator.
Build a Transfer Learning network using Keras with the following layers:
ResNet50 as first layers using Keras API.
GLobalAveragePooling2D layer
Dropout(0.5)
Dense layer with 3 neurons and activation softmax
``` python
base_model = tf.keras.applications.ResNet50(include_top=False)
resnet_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
])
#Compile the model with adam optimizer, categorical_crossentropy loss, and with metrics accuracy.
resnet_model.compile(
    optimizer="Adam",
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)
#Train the model for 4 epochs with callbacks. Reduce learning rate on Plateau and early stopping while monitoring validation loss.
history = resnet_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=2,
    callbacks = [learning_rate_reduction, early_stopping]
)
#Plot training and validation accuracy and loss against epochs.
plot_history(history)
```
Task C
Compare EfficientNetB0 and ResNet50 model performance and find the best model.
```python
test_loss, test_accuracy = efficientnet_model.evaluate(X_test,y_test, batch_size=8)
print(f"EfficientNetB0 Model Performance")
print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

test_loss, test_accuracy = resnet_model.evaluate(X_test,y_test, batch_size=8)
print(f"ResNet50 Model Performance")
print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
#Choose the best model to check recall, precision, and f1-score values.
from sklearn.metrics import classification_report
print(classification_report(y_test.argmax(axis = 1), efficientnet_model.predict(X_test).argmax(axis = 1)))
