import neat
import keras 
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint 
import keras_tuner
from keras_tuner import HyperModel, RandomSearch
import pandas as pd 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random 
import sklearn 
from sklearn import preprocessing, model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

# Preprocessing Continued...

train_data = pd.read_csv("training_data.csv")
val_data = pd.read_csv("validation_data.csv")
test_data = pd.read_csv("testing_data.csv")

# Drop the date column from each dataset
train_data = train_data.drop("Date", axis=1)
val_data = val_data.drop("Date", axis=1)
test_data = test_data.drop("Date", axis=1)

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# Split each dataset into input features (X) and target outputs (y)
X_train = train_data_scaled[:, :7]
y_train = train_data_scaled[:, 7:]

X_val = val_data_scaled[:, :7]
y_val = val_data_scaled[:, 7:]

X_test = test_data_scaled[:, :7]
y_test = test_data_scaled[:, 7:]

# Reshape the input data to match the expected input shape for the model
num_features = X_train.shape[1]


X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)


X_train = X_train.reshape((X_train.shape[0], 1, num_features))
X_val = X_val.reshape((X_val.shape[0], 1, num_features))
X_test = X_test.reshape((X_test.shape[0], 1, num_features))


#Define the initial Neural Network Architecture

# Define the input shape
input_shape = (None, 7)  # inputs are OHLC RSI ATR SLD

# Define the output shape
output_shape = 2  # 2 for Long_Outcome and Short_Outcome

# Define the neural network architecture

model = keras.Sequential(
    [
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(output_shape, activation="sigmoid"),
    ]
)


# Compile the model with an optimizer, loss function, and metric

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Print the model summary
model.summary()

# Training the model

# Specify the number of epochs and batch size
epochs = 20
batch_size = 64

# Specify the validation data for the training process
validation_data = (X_val, y_val)

# Implement early stopping to prevent overfitting and save the best performing model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=validation_data,
    callbacks=[es, mc]
)