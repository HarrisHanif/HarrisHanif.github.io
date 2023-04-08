import pandas as pd 
import numpy as np 
import gym 
import tensorflow as tf 
import keras 
import keras_tuner
from keras_tuner import HyperModel, RandomSearch
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
import netron 





def build_model(hp):
    model = Sequential()
    
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    layer_types = ['lstm', 'dense', 'conv1d']
    activations = ['relu', 'tanh', 'softmax']
    
    for i in range(num_layers):
        layer_type = hp.Choice(f'layer_type_{i}', layer_types)

        if layer_type == 'lstm':
            model.add(LSTM(units=hp.Int(f'lstm_units_{i}', min_value=16, max_value=128, step=16), 
                           activation=hp.Choice(f'lstm_activation_{i}', activations), 
                           return_sequences=i < num_layers - 1))
        elif layer_type == 'dense':
            model.add(Dense(units=hp.Int(f'dense_units_{i}', min_value=16, max_value=128, step=16), 
                            activation=hp.Choice(f'dense_activation_{i}', activations)))
        elif layer_type == 'conv1d':
            if i == 0:  # Add input shape for the first layer
                model.add(Conv1D(filters=hp.Int(f'conv1d_filters_{i}', min_value=16, max_value=128, step=16),
                                 kernel_size=hp.Int(f'conv1d_kernel_size_{i}', min_value=2, max_value=5, step=1),
                                 activation=hp.Choice(f'conv1d_activation_{i}', activations),
                                 input_shape=(7, 1)))
            else:
                model.add(Conv1D(filters=hp.Int(f'conv1d_filters_{i}', min_value=16, max_value=128, step=16),
                                 kernel_size=hp.Int(f'conv1d_kernel_size_{i}', min_value=2, max_value=5, step=1),
                                 activation=hp.Choice(f'conv1d_activation_{i}', activations)))
            model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # Increase the number of trials to explore more configurations
    executions_per_trial=3,  # Run each trial multiple times and average the results
    directory='random_search_logs',
    project_name='gold_trading'
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