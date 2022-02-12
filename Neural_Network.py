import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

# Load image dataset 
fashion_mnist = keras.datasets.fashion_mnist
# Split data into testing and training tuples 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define Array of clothing items 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Display sample pics 
'''
plt.figure()
plt.imshow(train_images[1], cmap = 'gray')
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Data Preprocessing -- Getting greyscale pixel values between 0 and 1
train_images = train_images / 255.0 
test_images = test_images / 255.0

# Build Model -- AKA Architecture of Model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)), # Input Layer
    keras.layers.Dense(128, activation = 'relu'), # Hidden Layer
    keras.layers.Dense(10, activation = 'softmax') # Output Layer
])

# Compile Model -- Hyperparameter Tuning
model.compile(optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])

# Train Model 
model.fit(train_images, train_labels, epochs = 10)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
print('Test Accuracy: ', test_acc) 

# Make Predictions using test data
predictions = model.predict(test_images)