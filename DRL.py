import pandas as pd 
import numpy as np 
import gym 
import tensorflow as tf 
import keras 
import keras_tuner
from keras_tuner import HyperModel, RandomSearch
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import datetime 
import netron 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import pydot
import graphviz 



def weighted_f1_score(y_true, y_pred):
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)
    y_true_class = K.eval(y_true_class)
    y_pred_class = K.eval(y_pred_class)

    cm = confusion_matrix(y_true_class, y_pred_class)
    weights = cm.sum(axis=1) / cm.sum()
    f1_scores = f1_score(y_true_class, y_pred_class, average=None)

    return np.average(f1_scores, weights=weights)

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

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=3)
        self.true_positives.assign_add(tf.cast(tf.linalg.diag_part(cm), tf.float32))
        self.false_positives.assign_add(tf.reduce_sum(cm, axis=0) - tf.linalg.diag_part(cm))
        self.false_negatives.assign_add(tf.reduce_sum(cm, axis=1) - tf.linalg.diag_part(cm))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return tf.reduce_mean(f1)

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

      


def build_model(hp):
    num_actions = 3  # Set the number of actions: long, short, do nothing

    model = keras.Sequential()

    # Add LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(keras.layers.LSTM(units=hp.Int('lstm_units_' + str(i), min_value=32, max_value=128, step=32),
                                    return_sequences=True if i < hp.Int('num_lstm_layers', 1, 3) - 1 else False,
                                    activation='relu'))
    
    # Add Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 4)):
        model.add(keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=32, max_value=128, step=32),
                                     activation='relu'))
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
    
    # Add output layer
    model.add(keras.layers.Dense(units=num_actions, activation='softmax'))  # Change activation to 'softmax'
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
    loss="sparse_categorical_crossentropy",
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), weighted_f1_score(num_classes=3, name="f1_score")],
)

    

    return model




tuner = RandomSearch(
    build_model,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    max_trials=2,
    executions_per_trial=3,
    directory='output',
    project_name='DRL',
    overwrite=True,
    seed=42,
)

tuner.search(
    X_train,
    y_train,
    epochs=20,  # You can adjust this based on your computational resources
    validation_split=0.2,
)

# Get the best hyperparameters found by the tuner
best_hp = tuner.get_best_hyperparameters(1)[0]

if tuner.oracle.get_best_trials(num_trials=1)[0].score is not None:
    best_model = tuner.get_best_models(num_models=1)[0]
else:
    print("No successful trials")


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
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_f1_score", patience=5, mode="max", restore_best_weights=True)]

)

# Load the best model
best_model = load_model("best_model.h5")


# Start the Netron server and open the model
netron.start("best_model.h5")

