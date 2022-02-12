# Predicting likelihood of survival on Titanic 
# 1 is survived -- 0 is not survived 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.python.feature_column.feature_column_v2 import NumericColumn

# Load Data
dftrain = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training Data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing Data
y_train = dftrain.pop('survived') # Labels for Training Data
y_eval = dfeval.pop('survived') # Labels for Testing Data 

# Define Categorical vs. Numeric Data 
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = [] # Store feature columns to feed into Linear Model/Estimator
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # Gets every unique value for each catagorical feature 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

# Input Function 
def make_input_fn(data_df, label_df, num_epochs = 22, shuffle = True, batch_size = 64): # Input fucntion to turn raw pandas data into Tensors that can be used by TensorFlow
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # Create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000) # Randomize order of data 
        ds = ds.batch(batch_size).repeat(num_epochs) # Split dataset into batches of 32 and repeat process for number of epochs 
        return ds
    return input_function 

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval)

linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns) # Creates the Linear Model/Esitmator using the feature columns as data input 

linear_est.train(train_input_fn) # Trains the model 
result = linear_est.evaluate(eval_input_fn) # Gets model stats by testing on testing data

print(result) # Result variable is a dict of all stats about model

person = 5 # Index of person in prediction list

result = list(linear_est.predict(eval_input_fn)) # Predicting survivability on eval data 
print(dfeval.loc[person]) # Printing Persons info 
result2 = result[person]['probabilities'][1] # Getting survival chance from predicted list 
percentage = "{:.0%}".format(result2) # Formatting survival chance as percent 
print(percentage) # Printing survival chance 
print(y_eval.loc[person]) # Printing actaul survival 

