import pandas as pd
import numpy as np
import gym
import tensorflow as tf
import keras
import keras_tuner
from keras import layers
from keras.layers import Rescaling, Layer
from keras_tuner import HyperModel, RandomSearch, Objective, BayesianOptimization, Hyperband
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Input, MultiHeadAttention
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.regularizers import l2
from keras import backend as K
import datetime
import netron
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix
import pydot
import os
import matplotlib.pyplot as plt
import seaborn as sns 


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

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def create_sliding_window_dataset(data, window_size):
    data_windowed = np.zeros((data.shape[0] - window_size, window_size, data.shape[1], data.shape[2]))
    for i in range(data.shape[0] - window_size):
        data_windowed[i] = data[i:i + window_size, :, :]
    return data_windowed


window_size = 20  #consider the previous X days

X_train_windowed = create_sliding_window_dataset(X_train, window_size)
X_val_windowed = create_sliding_window_dataset(X_val, window_size)
X_test_windowed = create_sliding_window_dataset(X_test, window_size)

y_train_windowed = y_train[window_size:]
y_val_windowed = y_val[window_size:]
y_test_windowed = y_test[window_size:]

class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f' - val_f1: {f1:.4f}')
        logs['val_f1'] = f1


EPOCHS=250

class F1ScoreObjective(Objective):
    def __init__(self):
        super(F1ScoreObjective, self).__init__('val_f1', 'max')

    def transform(self, value):
        return value



def build_model(hp):
    num_actions = 3

    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(window_size, 7), dtype="float32"))


    l2_lambda_value = hp.Float('l2_lambda_value', min_value=1e-6, max_value=1e-3, sampling='LOG')
    
    length_scale = hp.Float('length_scale', min_value=1e-6, max_value=1e9, sampling='LOG')  # Increased upper bound

    
    for i in range(hp.Int('num_lstm_layers', 3, 4)):  # Changed from 20-100 to 1-5
        model.add(keras.layers.LSTM(units=hp.Int('lstm_units_' + str(i), min_value=256, max_value=768, step=64),  # Changed max_value from 1024 to 512
                                    return_sequences=True if i < hp.Int('num_lstm_layers', 3, 4) - 1 else False,
                                    activation='tanh',
                                    kernel_regularizer=l2(l2_lambda_value),
                                    recurrent_regularizer=l2(l2_lambda_value),
                                    bias_regularizer=l2(l2_lambda_value)))
    
    for i in range(hp.Int('num_dense_layers', 1, 1)):  # Changed from 20-100 to 1-5
        model.add(keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=64, max_value=512, step=64),  # Changed max_value from 1024 to 512
                                 activation='softmax',
                                 kernel_regularizer=l2(l2_lambda_value)))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))

    model.add(keras.layers.Dense(units=num_actions, activation='softmax'))
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', f1])

    f1_callback = F1ScoreCallback((X_val_windowed, y_val_windowed))
    model.fit(
        X_train_windowed,
        y_train_windowed,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5), f1_callback],
        validation_data=(X_val_windowed, y_val_windowed),
        verbose=2
    )


    return model


project_dir = 'output'
project_name = 'DRL'


# Define the BayesianOptimization tuner
bayesian_tuner = BayesianOptimization(
    build_model,
    objective=F1ScoreObjective(),
    max_trials=25,
    num_initial_points=10,
    directory='my_dir_bayesian01',
    project_name='DRL5_bayesian01',
    seed=69
)




# Search using BayesianOptimization
bayesian_tuner.search(X_train_windowed, y_train_windowed, epochs=250, validation_data=(X_val_windowed, y_val_windowed), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=10, mode="max")])

# Get best BayesianOptimization model
best_hyperparameters_bayesian = bayesian_tuner.get_best_hyperparameters(num_trials=1)[0]




def ensemble_predictions(models, x_data):
    predictions = np.zeros((x_data.shape[0], len(models), 3))
    for i, model in enumerate(models):
        predictions[:, i, :] = model.predict(x_data)

    final_predictions = np.mean(predictions, axis=1)
    return final_predictions


k = 5
# Load the top k models
top_k_models = bayesian_tuner.get_best_models(num_models=k)
# Make predictions with the ensemble
y_pred = ensemble_predictions(top_k_models, X_test_windowed)



# Convert y_pred from probabilities to binary format
y_pred_binary = np.argmax(y_pred, axis=1)

# Convert y_test_windowed from multilabel-indicator to binary format
y_test_binary = np.argmax(y_test_windowed, axis=1)

# Evaluate the ensemble's performance
from sklearn.metrics import accuracy_score, f1_score

ensemble_accuracy = accuracy_score(y_test_binary, y_pred_binary)
ensemble_f1 = f1_score(y_test_binary, y_pred_binary, average='weighted')
print(f"Ensemble Accuracy: {ensemble_accuracy}")
print(f"Ensemble F1 Score: {ensemble_f1}")

# Calculate the confusion matrix itself
confusion = confusion_matrix(y_test_binary, y_pred_binary)

# Display the confusion matrix using Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()

model_checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_f1",
    save_best_only=True,
    mode="max",
    verbose=1,
)

# Get the best hyperparameters
best_hyperparameters = bayesian_tuner.get_best_hyperparameters(num_trials=1)[0]

model_checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_f1",
    save_best_only=True,
    mode="max",
    verbose=1,
)

best_model = bayesian_tuner.hypermodel.build(best_hyperparameters)
history = best_model.fit(
    X_train_windowed,
    y_train_windowed,
    epochs=250,
    validation_data=(X_val_windowed, y_val_windowed),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=10, mode="max"), model_checkpoint]
)

# Plotting loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.show()


best_model = load_model("best_model.h5", custom_objects={'f1': f1})



netron.start("best_model.h5")
