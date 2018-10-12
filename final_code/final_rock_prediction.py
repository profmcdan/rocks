
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)


# In[9]:


dataset = np.loadtxt("DataNN_formatted.csv", delimiter=",")
# split into input (X) and output (Y) variables
train_data = dataset[:,0:16]
train_labels = dataset[:,16]


# In[12]:


print("Training set: {}".format(train_data.shape))  # 9 examples, 16 features
# print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

"""The dataset contains 16 different features:

Compressive Strength MPa.
Aggregate Crushing Value.
Specific Gravity.
Water Absorption.
Point Load.
Brittleness Value.
Siever J (SJ) Value.
Density.
Location.
Dynamic Young Modulus.
Dynamic Shear Modulus.
P – Wave Velocity (m/s).
S – Wave Velocity (m/s).
Poission’s Ratio (V).
Rock Hardness (f).
Drilling Rate Index (DRI).


Each one of these input data features is stored using a different scale. Some features are represented by a proportion between 0 and 1, other features are ranges between 1 and 12, some are ranges between 0 and 100, and so on. This is often the case with real-world data, and understanding how to explore and clean such data is an important skill to develop.

Key Point: As a modeler and developer, think about how this data is used and the potential benefits and harm a model's predictions can cause. A model like this could reinforce societal biases and disparities. Is a feature relevant to the problem you want to solve or will it introduce bias? For more information, read about ML fairness.
# In[13]:
"""


print(train_data[0])  # Display sample features, notice the different scales


# In[18]:


import pandas as pd

column_names = ['Compressive Strength MPa', 'Aggregate Crushing Value', 'Specific Gravity', 'Water Absorption', 'Point Load', 'Brittleness Value', 'Siever J (SJ) Value', 'Density', 'Location',
                'Dynamic Young Modulus', 'Dynamic Shear Modulus', 'P – Wave Velocity (m/s)', 'S – Wave Velocity (m/s)', 'Poission’s Ratio (V)'
               'Rock Hardness (f)', 'Drilling Rate Index (DRI)'], 

# df = pd.DataFrame(train_data, columns=column_names)
df = pd.DataFrame(train_data)
df.head()


# In[19]:


print(train_labels[0:10])


# ## Normalize features¶
# It's recommended to normalize features that use different scales and ranges. For each feature, subtract the mean of the feature and divide by the standard deviation:

# In[21]:


# Test data is *not* used when calculating the mean and std
test_data = train_data

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized


# Although the model might converge without feature normalization, it makes training more difficult, and it makes the resulting model more dependent on the choice of units used in the input.
# 
# ## Create the model
# Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and an output layer that returns a single, continuous value. The model building steps are wrapped in a function, build_model, since we'll create a second model, later on.

# In[22]:


def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()


# ## Train the model¶
# The model is trained for 500 epochs, and record the training and validation accuracy in the history object.

# In[23]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])


# Visualize the model's training progress using the stats stored in the history object. We want to use this data to determine how long to train before the model stops making progress.

# In[26]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [DRI]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

plot_history(history)


# 
# This graph shows little improvement in the model after about 200 epochs. Let's update the model.fit method to automatically stop training when the validation score doesn't improve. We'll use a callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.

# In[27]:



model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)


# The graph shows the average error is about xx. Is this good? Well, xx is not an insignificant amount when some of the labels are only yy.
# 
# Let's see how did the model performs on the test set:

# In[30]:


test_labels = train_labels
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae))





# ## Predict

# In[31]:



test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [DRI]')
plt.ylabel('Predictions [DRI]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])


# In[32]:


error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [DRI]")
_ = plt.ylabel("Count")


# 
# ## Conclusion
# This notebook introduced a few techniques to handle a regression problem.
# 
# Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
# Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
# When input data features have values with different ranges, each feature should be scaled independently.
# If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
# Early stopping is a useful technique to prevent overfitting.
