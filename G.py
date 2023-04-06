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
import netron 


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
        layers.GRU(64, input_shape=input_shape, return_sequences=True),
        layers.LSTM(64),
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

class GoldTradingHyperModel(HyperModel):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self, hp):
        model = keras.Sequential(
            [
                layers.GRU(
                    units=hp.Int("gru_units", min_value=32, max_value=128, step=32),
                    input_shape=self.input_shape,
                    return_sequences=True,
                ),
                layers.LSTM(
                    units=hp.Int("lstm_units", min_value=32, max_value=128, step=32)
                ),
                layers.Dense(
                    units=hp.Int("dense_units", min_value=32, max_value=128, step=32),
                    activation="relu",
                ),
                layers.Dropout(
                    hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
                ),
                layers.Dense(self.output_shape, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model


timesteps, num_features = X_train.shape[1], X_train.shape[2]
input_shape = (timesteps, num_features)


input_shape = (timesteps, 7)  
output_shape = 2 
hypermodel = GoldTradingHyperModel(input_shape, output_shape)


tuner = RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=200,  # You can adjust this based on your computational resources
    executions_per_trial=5,
    directory="gold_trading",
    project_name="gold_trading_hyperparam_tuning",
)

tuner.search(
    X_train,
    y_train,
    epochs=20,  # You can adjust this based on your computational resources
    validation_split=0.2,
)

best_model = tuner.get_best_models(num_models=1)[0]

# Save the best model's architecture and weights to a file
best_model.save("best_model.h5")

# Start the Netron server and open the model
netron.start("best_model.h5")
