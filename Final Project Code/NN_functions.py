import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.layers import concatenate as Concatenate
from tensorflow.keras.optimizers import Adam

def import_data():
       NN_data = pd.read_csv("Data/NNData.csv")
       X = NN_data.iloc[:,3:]
       y = NN_data.iloc[:,2]
       return X, y


def split_data(X, y, random_state):
       # Split
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

       # Separate linear and sigmoid variables
       X_train_lin = X_train[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Monday', 'Tuesday', 'Wednesday',
       'Thursday', 'Friday', 'Saturday', 'BankHols', 'RetailHols',
       'WkBeforeXMas', 'WkAfterXMas', 'Trend2010']]
       X_test_lin = X_test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Monday', 'Tuesday', 'Wednesday',
       'Thursday', 'Friday', 'Saturday', 'BankHols', 'RetailHols',
       'WkBeforeXMas', 'WkAfterXMas', 'Trend2010']]
       X_train_sig = X_train[['Trend2010', 'AvgDB', 'LagDB', 'Lag2DB',
       'AvgWind', 'AvgClouds']]
       X_test_sig = X_test[['Trend2010', 'AvgDB', 'LagDB', 'Lag2DB',
       'AvgWind', 'AvgClouds']]

       # Change datatype
       X_train_lin = np.asarray(X_train_lin).astype('float32')
       X_test_lin = np.asarray(X_test_lin).astype('float32')
       X_train_sig = np.asarray(X_train_sig).astype('float32')
       X_test_sig = np.asarray(X_test_sig).astype('float32')
       y_train = np.asarray(y_train).astype('float32')
       y_test = np.asarray(y_test).astype('float32')

       # Scale data
       scaler_x_lin = StandardScaler()
       X_train_lin = scaler_x_lin.fit_transform(X_train_lin)
       X_test_lin = scaler_x_lin.transform(X_test_lin)
       
       scaler_x_sig = StandardScaler()
       X_train_sig = scaler_x_sig.fit_transform(X_train_sig)
       X_test_sig = scaler_x_sig.transform(X_test_sig)

       scaler_y = MinMaxScaler()
       y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).reshape(-1,)

       return X_train_lin, X_test_lin, X_train_sig, X_test_sig, y_train, y_test, scaler_y


def create_model(nodes):
       # Define input shapes based on number of variables
       input_shape_1 = Input(shape=(22,))
       input_shape_2 = Input(shape=(6,))
       # Linear node
       tower_1 = Dense(1)(input_shape_1)
       tower_1 = Activation(activation="linear")(tower_1)
       # Sigmoid nodes
       tower_2 = Dense(nodes)(input_shape_2)
       tower_2 = Activation(activation="sigmoid")(tower_2)
       # Combine the nodes
       merged = Concatenate([tower_1, tower_2], axis=1)
       merged = Flatten()(merged)
       out = Dense(1)(merged)
       out = Activation(activation="linear")(out)
       # Create model
       model = Model([input_shape_1, input_shape_2], out)
       # Defin the optimizer
       optimizer = Adam(clipnorm = 1)
       # Compile the model
       model.compile(optimizer=optimizer, loss=["MSE"], metrics=["MAE", "MAPE"])

       return model