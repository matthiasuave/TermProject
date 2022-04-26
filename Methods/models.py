import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, SimpleRNN
import data_prep


def root_mean_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred)))

## we should consider normalizing the data

class RunModel:

    def __init__(self,X_train, X_test, y_train, y_test):
        self.X_train = tf.convert_to_tensor(X_train)
        self.y_train = tf.convert_to_tensor(y_train)
        self.X_test = tf.convert_to_tensor(X_test)
        self.y_test = tf.convert_to_tensor(y_test)


    def rnn_model(self):
        model_rnn = Sequential()
        model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(6,1)))
        model_rnn.add(Dense(1))
        print('\nRunning RNN model...')
        model_rnn.compile(optimizer='adam', loss='mse', metrics='mape')
        model_rnn.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2, batch_size=100)
        
        train_loss, train_mape = model_rnn.evaluate(self.X_train, self.y_train)
        print(f'RNN Model: \nTraining set has a loss (MSE) of {train_loss} with Mean Absolute Percentage Error (MAPE) of {train_mape}')

        test_loss, test_mape = model_rnn.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with Mean Absolute Percentage Error (MAPE) of {test_mape}\n')


    def lstm_model(self):
        model_lstm = Sequential()
        model_lstm.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(6,1)))
        model_lstm.add(Dense(1))
        print('\nRunning the LSTM model...')
        model_lstm.compile(optimizer='adam', loss='mse', metrics='mape')
        model_lstm.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2, batch_size=100)
        
        train_loss, train_mape = model_lstm.evaluate(self.X_train, self.y_train)
        print(f'LSTM Model: \nTraining set has a loss (MSE) of {train_loss} with Mean Absolute Percentage Error (MAPE) of {train_mape}')

        test_loss, test_mape = model_lstm.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with Mean Absolute Percentage Error (MAPE) of {test_mape}\n')


    def gru_model(self):
        model_gru = Sequential()
        model_gru.add(GRU(50, activation='relu', input_shape=(6,1)))
        model_gru.add(Dense(1))
        print('\nRunning RNN model...')
        model_gru.compile(optimizer='adam', loss='mse', metrics='mape')
        model_gru.fit(self.X_train, self.y_train, epochs=5, validation_split=0.2, batch_size=100)
        
        train_loss, train_mape = model_gru.evaluate(self.X_train, self.y_train)
        print(f'GRU Model: \nTraining set has a loss (MSE) of {train_loss} with Mean Absolute Percentage Error (MAPE) of {train_mape}')

        test_loss, test_mape = model_gru.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with Mean Absolute Percentage Error (MAPE) of {test_mape}\n')


    def run_all_models(self):
        self.rnn_model()
        self.lstm_model()
        self.gru_model()


if __name__ == '__main__':
    clean_data = data_prep.DataCleaning()
    X_train, X_test, y_train, y_test = clean_data.SampleValidSequences(numTrainSequences=200, numTestSequences=20)

    model_obj = RunModel(X_train, X_test, y_train, y_test)
    model_obj.run_all_models()