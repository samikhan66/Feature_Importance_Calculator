import os
import random
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


# ToDO: Implement AutoML to select the best Neural Network Architecture and Optimization algorithms
class DNN():
    def __init__(self, X_train, y_train):
        """Instantiate X_train and y_train"""
        self.X_train = X_train
        self.y_train = y_train
        # Set random seed to 42 so that the same set is selected every time
        tf.random.set_seed(42)
        os.environ['PYTHONHASHSEED'] = str(42)
        np.random.seed(42)
        random.seed(42)
        # Build tensorflow session
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1
        )
        self.sess = tf.compat.v1.Session(
            graph=tf.compat.v1.get_default_graph(),
            config=session_conf
        )
        tf.compat.v1.keras.backend.set_session(self.sess)

    def build(self):
        """Build the neural network architecture"""
        # Scale variables
        scaler = StandardScaler()
        self.scaled_train = scaler.fit_transform(self.X_train)

        # Define NN structure
        inp = Input(shape=(self.scaled_train.shape[1],))
        x = Dense(500, activation='relu')(inp)
        x = Dense(500, activation='relu')(inp)
        x = Dense(500, activation='relu')(inp)
        x = Dense(250, activation='relu')(inp)
        x = Dense(25, activation='relu')(x)
        out = Dense(1)(x)
        self.model = Model(inp, out)

    def compile_and_fit(self):
        """Compile and fit the model"""
        mean_train = self.y_train.mean()
        std_train = self.y_train.std()
        self.model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
        self.model.fit(self.scaled_train, scale_target(self.y_train, mean_train, std_train), epochs=500, batch_size=64,
                       verbose=2)

    def compute_scores(self, X_test, y_test):
        # Scale variables
        scaler = StandardScaler()
        scaled_test = scaler.fit_transform(X_test)

        # Compute permutation and scoring
        os.environ['PYTHONHASHSEED'] = str(42)
        np.random.seed(42)
        random.seed(42)
        mean_train = self.y_train.mean()
        std_train = self.y_train.std()

        final_score = []
        shuff_pred = []

        for i, col in enumerate(X_test.columns):
            # shuffle column
            shuff_test = scaled_test.copy()
            shuff_test[:, i] = np.random.permutation(shuff_test[:, i])

            # compute score
            score = mean_absolute_error(y_test,
                                        reverse_target(self.model.predict(shuff_test).ravel(), mean_train, std_train))
            shuff_pred.append(reverse_target(self.model.predict(shuff_test).ravel(), mean_train, std_train))
            final_score.append(score)

        final_score = np.asarray(final_score)

        return final_score


def scale_target(y, mean, std):
    """Use to scale the target variable using the formula => (value - mean)/standard deviation.
    We need to performing this scaling in order to feed it while training the Neural Network"""
    return np.asarray((y - mean) / std)


def reverse_target(pred, mean, std):
    """Use to reverse the scaled target variable using the formula => Predicted_value*standard deviation + mean
    # We need to perform this rescaling in order to compare the predictions"""
    return np.asarray(pred * std + mean)
