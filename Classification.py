# Use properties of flowers to predict specoes of flower 

import tensorflow as tf 
import pandas as pd
from tensorflow.core.example.feature_pb2 import Features 

# Data Constants
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Use Keras to load data 
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# Convert Keras data to Pandas dataframe 
train = pd.read_csv(train_path, names = CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header = 0)

# Pop Label data
train_y = train.pop('Species')
test_y = test.pop('Species')

# Input Function 
def input_fn(features, labels, training = True, batch_size = 256):
    # Convert data into Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffles Data + Repeat if in Training Mode 
    if training: 
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

# Define Feature Column 
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key = key))
print(my_feature_columns
)

# DNN with 2 hidden layers -- layer 1 30 nodes -- layer 2 10 nodes
classifer = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    # Two hidden layers
    hidden_units = [30, 10],
    # Model must choose between 3 classes
    n_classes = 3
)

# Train Model 
classifer.train(
    # lambda to create one line function so that we can have a function object
    input_fn = lambda: input_fn(train, train_y, training = True),
    steps = 5000
)
# Evaluate Model 
eval_result = classifer.evaluate(input_fn = lambda: input_fn(test, test_y, training = False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Use model to predict on one value
def predict_input_fn(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features =  ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Input values")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ":")
        if not val.isdigit(): 
            valid = False
    
    predict[feature] = [float(val)]

predictions = classifer.predict(input_fn = lambda: predict_input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))