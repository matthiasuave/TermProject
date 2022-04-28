from cProfile import label
from statistics import mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Flatten



def root_mean_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(y_true, y_pred)))


class RunModel:

    def __init__(self,X_train, X_test, y_train, y_test):
        #self.X_train = (tf.convert_to_tensor(X_train) - tf.math.reduce_min(tf.convert_to_tensor(X_train))) / (tf.math.reduce_max(tf.convert_to_tensor(X_train)) - tf.math.reduce_min(tf.convert_to_tensor(X_train)))
        #self.y_train = (tf.convert_to_tensor(y_train) - tf.math.reduce_min(tf.convert_to_tensor(y_train))) / (tf.math.reduce_max(tf.convert_to_tensor(y_train)) - tf.math.reduce_min(tf.convert_to_tensor(y_train)))
        #self.X_test = (tf.convert_to_tensor(X_test) - tf.math.reduce_min(tf.convert_to_tensor(X_test))) / (tf.math.reduce_max(tf.convert_to_tensor(X_test)) - tf.math.reduce_min(tf.convert_to_tensor(X_test)))
        #self.y_test = (tf.convert_to_tensor(y_test) - tf.math.reduce_min(tf.convert_to_tensor(y_test))) / (tf.math.reduce_max(tf.convert_to_tensor(y_test)) - tf.math.reduce_min(tf.convert_to_tensor(y_test)))
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.metric_df = pd.DataFrame()

    def rnn_model(self):
        model_rnn = Sequential()
        model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(8,1)))
        model_rnn.add(Dense(10))
        model_rnn.add(Dense(1))
        print('\nRunning RNN model...')
        model_rnn.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        hist = model_rnn.fit(self.X_train, self.y_train, epochs=7, validation_split=0.2, batch_size=100)
        
        train_loss, train_rmse = model_rnn.evaluate(self.X_train, self.y_train)
        print(f'RNN Model: \nTraining set has a loss (MSE) of {train_loss} with RMSE metric of {train_rmse}')

        test_loss, test_rmse = model_rnn.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with RMSE metric of {test_rmse}\n')
        y_pred = model_rnn.predict(self.X_test)
        plt.plot(range(len(y_pred)),y_pred, label='Prediction')
        plt.plot(self.y_test, label='Actual')
        plt.xlabel('Time Series')
        plt.ylabel('Readings')
        plt.title('Simple RNN MODEL')
        plt.legend()
        plt.savefig('Model Fig - SimpleRNN.png')
        plt.clf()
        model_rnn.summary()
        self.metric_df['RNN RMSE'] = hist.history['val_root_mean_squared_error']



    def lstm_model(self):
        model_lstm = Sequential()
        model_lstm.add(LSTM(128, activation='relu', input_shape=(8, 1),return_sequences=True))
        model_lstm.add(Dropout(0.3))
        model_lstm.add(LSTM(64, activation='relu'))
        model_lstm.add(Dropout(0.3))
        model_lstm.add(Dense(20, activation='relu'))
        model_lstm.add(Dense(10, activation='relu'))
        model_lstm.add(Dense(1))
        print('\nRunning the LSTM model...')
        model_lstm.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        hist = model_lstm.fit(self.X_train, self.y_train, epochs=7, validation_split=0.2, batch_size=100)
        
        train_loss, train_rmse = model_lstm.evaluate(self.X_train, self.y_train)
        print(f'LSTM Model: \nTraining set has a loss (MSE) of {train_loss} with RMSE metric of {train_rmse}')

        test_loss, test_rmse = model_lstm.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with RMSE metric of {test_rmse}\n')
        y_pred = model_lstm.predict(self.X_test)
        plt.plot(range(len(y_pred)),y_pred, label='Prediction')
        plt.plot(self.y_test, label='Actual')
        plt.xlabel('Time Series')
        plt.ylabel('Readings')
        plt.title('LSTM MODEL')
        plt.legend()
        plt.savefig('Model Fig - LSTM.png')
        plt.clf()
        model_lstm.summary()
        self.metric_df['LSTM RMSE'] = hist.history['val_root_mean_squared_error']


    def gru_model(self):
        model_gru = Sequential()
        model_gru.add(GRU(50, activation='relu', input_shape=(8,1)))
        model_gru.add(Dense(1))
        print('\nRunning GRU model...')
        model_gru.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        hist = model_gru.fit(self.X_train, self.y_train, epochs=7, validation_split=0.2, batch_size=100)
        
        train_loss, train_rmse = model_gru.evaluate(self.X_train, self.y_train)
        print(f'GRU Model: \nTraining set has a loss (MSE) of {train_loss} with RMSE metric of {train_rmse}')

        test_loss, test_rmse = model_gru.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with RMSE metric of {test_rmse}\n')
        y_pred = model_gru.predict(self.X_test)
        plt.plot(range(len(y_pred)),y_pred, label='Prediction')
        plt.plot(self.y_test, label='Actual')
        plt.xlabel('Time Series')
        plt.ylabel('Readings')
        plt.title('GRU MODEL')
        plt.legend()
        plt.savefig('Model Fig - GRU.png')
        plt.clf()
        model_gru.summary()
        self.metric_df['GRU RMSE'] = hist.history['val_root_mean_squared_error']


    def cnn_lstm_model(self):
        model_cnn_lstm = Sequential()
        model_cnn_lstm.add(tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(8,1)))
        model_cnn_lstm.add(tf.keras.layers.MaxPooling1D((1)))
        model_cnn_lstm.add(LSTM(10, activation='relu', return_sequences=True))
        model_cnn_lstm.add(Flatten())
        model_cnn_lstm.add(Dense(1))
        print('\nRunning the CNN+LSTM model...')
        model_cnn_lstm.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        hist = model_cnn_lstm.fit(self.X_train, self.y_train, epochs=7, validation_split=0.2, batch_size=100)
        
        train_loss, train_rmse = model_cnn_lstm.evaluate(self.X_train, self.y_train)
        print(f'CNN+LSTM Model: \nTraining set has a loss (MSE) of {train_loss} with RMSE metric of {train_rmse}')

        test_loss, test_rmse = model_cnn_lstm.evaluate(self.X_test, self.y_test)
        print(f'Test set has a loss (MSE) of {test_loss} with RMSE metric of {test_rmse}\n')
        y_pred = model_cnn_lstm.predict(self.X_test)
        print(y_pred.shape)
        plt.plot(range(len(y_pred)),y_pred, label='Prediction')
        plt.plot(self.y_test, label='Actual')
        plt.xlabel('Time Series')
        plt.ylabel('Readings')
        plt.title('CNN+LSTM MODEL')
        plt.legend()
        plt.savefig('Model Fig - CNN+LSTM.png')
        plt.clf()
        model_cnn_lstm.summary()
        self.metric_df['CNN+LSTM RMSE'] = hist.history['val_root_mean_squared_error']


    def main(self):
        self.rnn_model()
        self.lstm_model()
        self.gru_model()
        self.cnn_lstm_model()
        self.metric_df.plot(xlabel='Epochs', ylabel='RMSE', legend=True)
        plt.show()
