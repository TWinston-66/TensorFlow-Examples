import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load Data 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to between 0 and 1 
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define Class Names 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Look at images
img_index = 1

#plt.imshow(train_images[img_index], cmap = plt.cm.binary)
#plt.xlabel(class_names[train_labels[img_index][0]])
#plt.show()

# Build Convolutional Base  
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

# Build Dense Classifier 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

# Compile Model 
model.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
               metrics = ['accuracy'])

# Train Model 
history = model.fit(train_images, train_labels, epochs = 10,
                    validation_data = (test_images, test_labels))

# Evaluate Model 
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)