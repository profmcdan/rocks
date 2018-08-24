# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("DataNN_formatted.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:16]
Y = dataset[:,16]
# create model
model = Sequential()
model.add(Dense(24, input_dim=16, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=500, batch_size=2)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
