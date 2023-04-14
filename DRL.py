import pandas as pd 
import numpy as np 
import gym 
import tensorflow as tf 
import keras 
import keras_tuner
from keras import layers
from keras.layers import Rescaling, Layer
from keras_tuner import HyperModel, RandomSearch
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Input, MultiHeadAttention 
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
import datetime 
import netron 
from sklearn.preprocessing import MinMaxScaler
import pydot
import os 





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

# Split input features (X) and target outputs (y)
X_train = train_data.drop(columns=['Long_Outcome', 'Short_Outcome', 'Do_Nothing']).values
y_train = train_data[['Long_Outcome', 'Short_Outcome', 'Do_Nothing']].values

X_val = val_data.drop(columns=['Long_Outcome', 'Short_Outcome', 'Do_Nothing']).values
y_val = val_data[['Long_Outcome', 'Short_Outcome', 'Do_Nothing']].values

X_test = test_data.drop(columns=['Long_Outcome', 'Short_Outcome', 'Do_Nothing']).values
y_test = test_data[['Long_Outcome', 'Short_Outcome', 'Do_Nothing']].values

# Reshape the input data to have a shape of (7, 1) to match the input shape required by the Conv1D layers
X_train = X_train.reshape(-1, 7, 1)
X_val = X_val.reshape(-1, 7, 1)
X_test = X_test.reshape(-1, 7, 1)



def build_model(hp):
    num_actions = 3  # Set the number of actions: long, short, do nothing

    model = keras.Sequential()

    # Add Input layer
    model.add(keras.layers.Input(shape=(7,1), dtype="float32"))

    # Add LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 7)):
        model.add(keras.layers.LSTM(units=hp.Int('lstm_units_' + str(i), min_value=64, max_value=256, step=32),
                                    return_sequences=True if i < hp.Int('num_lstm_layers', 1, 7) - 1 else False,
                                    activation='tanh'))  # Change activation to 'tanh'
    
    # Add Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=64, max_value=256, step=32),
                                     activation='relu'))
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
    
    # Add output layer
    model.add(keras.layers.Dense(units=num_actions, activation='softmax'))  # Change activation to 'softmax'
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='categorical_crossentropy',  # Change loss to 'categorical_crossentropy'
                  metrics=['categorical_accuracy'])  # Add categorical_accuracy as a metric

    return model


project_dir = 'output'
project_name = 'DRL'



tuner = RandomSearch(
    build_model,
    objective='val_categorical_accuracy',  # Change objective to 'val_categorical_accuracy'
    max_trials=20,
    executions_per_trial=3,
    directory=project_dir,
    project_name=project_name,
    overwrite=True,
    seed=42,
)

# Load existing trials and best hyperparameters
if os.path.exists(os.path.join(project_dir, project_name)):
    tuner.reload()
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Loaded best hyperparameters:", best_hp)
else:
    print("No previous trials found, starting new search.")
    best_hp = None

# Run the search only if no previous trials were found
if best_hp is None:
    tuner.search(
        X_train,
        y_train,
        epochs=100,
        validation_split=0.2,
    )


# Get the best hyperparameters found by the tuner
if best_hp is None:
    best_hp = tuner.get_best_hyperparameters(1)[0]


best_model = tuner.get_best_models(num_models=1)[0]

# Set up the ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

# Train the best model using the training and validation data
history = best_model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), model_checkpoint]
)

# Load the best model
best_model = load_model("best_model.h5")


# Start the Netron server and open the model
netron.start("best_model.h5")