from keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# unzip the file containing the images
!unzip '/content/Faces.zip'

import os

# path to the images files
folder_path = '/content/NoSunglasses'
# Get all files in the folder
files = os.listdir(folder_path)
# Get the full path of each file
nosun_paths = [os.path.join(folder_path, file) for file in files]


# path to the images files
folder_path = '/content/Sunglassess'
# Get all files in the folder
files = os.listdir(folder_path)
# Get the full path of each file
sun_paths = [os.path.join(folder_path, file) for file in files]


import numpy as np
import os
import tensorflow as tf
import pandas as pd
import cv2
IMG_WIDTH=32
IMG_HEIGHT=30

img_data=[]
class_name=[]

# Extract the image array and class name
for path in [nosun_paths, sun_paths]:
  for file in path:
    image = cv2.imread(file, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),
    interpolation = cv2.INTER_AREA)
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    img_data.append(image)
    class_name.append(str(path))

target_dict={k: v for v, k in enumerate(np.unique(class_name))}
# Convert the class_names to their respective numeric value based on the dictionary
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
# Convert to X - a 3d array (samples, width, height) and y - 0/1
X = tf.cast(np.array(img_data), tf.float64)
y = tf.cast(list(map(int,target_val)),tf.int32)

data = np.array(X)
labels = np.array(y)

# image size
data[0].shape

from sklearn.model_selection import train_test_split
# splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

 Verify data
class_names = ['No Sunglasses', 'Sunglasses']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])
plt.show()

import numpy as np
np.unique(y_train, return_counts=True)

# Create convolutional base
def create_cnn_with_classifier():
  model = models.Sequential()
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 30, 1)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  # Compile and train the model
  model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
  
  return model

# Create convolutional base
def create_cnn():
  model = models.Sequential()
  model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 30, 1)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1, activation='sigmoid'))
  
  return model

####CNN models with optimizer ADAM with different batch sizes and epochs

# create model
model = KerasClassifier(model=create_cnn_with_classifier, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 50, 60]
epochs = [5, 10, 20, 30]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results

print("best accuracy:" + str(grid_result.best_score_) + " using:" + str(grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Accuracy: %f, STD: %f with: %r" % (mean, stdev, param))

####CNN models with different optimizers

# create model
model = KerasClassifier(model=create_cnn , loss="binary_crossentropy", epochs=30, batch_size=10, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)

# summarize results

print("best accuracy:" + str(grid_result.best_score_) + " using:" + str(grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Accuracy: %f, STD: %f with: %r" % (mean, stdev, param))

###Evaluation metrics

# Creating the best model according our results
model = KerasClassifier(model=create_cnn_with_classifier, epochs=30, batch_size=10, verbose=0)
# Training the model on the train data
model.fit(x_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(x_test)

# Printing a Classification report for the test set
print(classification_report(y_test, y_pred))

# Creating Confusion Matrix
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(15, 15))
fig.set_size_inches(5, 5)
sns.heatmap(mat, annot=True, cmap='Blues',fmt='.0f')
plt.title("Confusion Matrix")
plt.ylabel('Actual label(0 - no sunglasses; 1 - sunglasses)')
plt.xlabel('Predicted label(0 - no sunglasses; 1 - sunglasses)')

plt.show()

history = create_cnn_with_classifier().fit(x_train, y_train, epochs=30, batch_size=10,validation_data=(x_test, y_test))

# Plot showing the Loss Score over the epochs
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss Score')
plt.title('Loss Score over the epochs')
plt.ylim([0, 1])
plt.legend(loc='upper right')

# Plot showing the Accuracy over the epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Score over the epochs')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

