from tkinter import Label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Concatenating the time columns to create a single time-stamp column
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    """
    Credit - https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    Frame a time series as a supervised learning dataset.

    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.

    Returns:
    Pandas DataFrame of series framed for supervised learning.

    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,seq_length):
        super(RNN, self).__init__()

        self.hidden_dim=hidden_dim
        self.n_time_stamps = seq_length

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(seq_length*hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        # shape output to the linear layer (batch_size, seq_length*hidden_dim)
        r_out = r_out.contiguous().view(batch_size,-1)

        # get final output
        output = self.fc(r_out)

        return output, hidden


def mainFunction():
    # Importing data file
    datafile = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv'
    df = pd.read_csv(datafile)
    
    # Parsing and concatenating date & time columns
    dataset = pd.read_csv(datafile,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    
    # Dropping and renaming columns
    dataset.drop("No", axis=1, inplace=True)
    dataset.columns = ['pollution', 'dew', 'temp', 'pressure', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'

    # Filling NaN values
    dataset['pollution'].fillna(0, inplace=True)
    dataset = dataset[24:]
    
    # Encoding
    values = dataset.values
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])
    values = values.astype('float32')
    
    # Normalizing data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledValues = scaler.fit_transform(values)
    print(scaledValues[1:5,:])

    reframed = series_to_supervised(scaledValues, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(reframed[1:5])

    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24 *4
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, sequence length, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # Converting to PyTorch Tensors
    batch_size = 256
    x_train = torch.tensor(train_X , dtype=torch.float)
    y_train = torch.tensor(train_y, dtype=torch.float)
    x_test = torch.tensor(test_X , dtype=torch.float)
    y_test = torch.tensor(test_y , dtype=torch.float)
    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    # Let's check the shape of the input/target data
    dataiter = iter(train_loader)
    data, target = next(dataiter)
    print(type(data))
    print(data.shape)
    print(target.shape)

    # Decide on parameters and call the network
    input_size=8
    output_size=1
    hidden_dim=512
    n_layers=1
    seq_length = 1

    # instantiate an RNN
    rnn = RNN(input_size, output_size, hidden_dim, n_layers , seq_length)
    rnn.cuda()
    print(rnn)

    # MSE loss and Adam optimizer with a learning rate of 0.0001
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)

    hidden = None # initial hidden
    for epoch in range(30): ## run the model for 30 epochs
        train_loss = []

        for data, target in train_loader:

            data = data.to(torch.device('cuda'))
            target = target.to(torch.device('cuda'))

            if data.shape[0] != batch_size:   # to verify if the batch no is 256 or not
                #print('Batch Size Validation- Input shape Issue:',format(data.shape))
                continue
            else:
                optimizer.zero_grad()
                ## 1. forward propagation
                prediction, hidden = rnn(data, hidden)


                ## Representing Memory ##
                # make a new variable for hidden and detach the hidden state from its history
                # this way, we don't backpropagate through the entire history
                hidden = hidden.data
                batch_size = data.shape[0]

                ## 2. loss calculation
                loss = criterion(prediction.squeeze(), target)    # squeeze (256,1) -> (256) - to match target shape

                ## 3. backward propagation
                loss.backward()

                ## 4. weight optimization
                optimizer.step()

                train_loss.append(loss.item())

        print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))


    # Prediction using tensor of predictors i.e x_test
    yhat , _ = rnn(x_test.to(torch.device('cuda')), None) # throwing away _ the hidden
    
    #need to convert yhat to numpy
    yhat = yhat.cpu()
    yhat = yhat.detach().numpy()
    
    # To invert scale we need to reshape the 3D array to 2D array
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat.reshape(-1,1), test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -7:]), axis=1) #Note: -7 to select correct inputs
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    from math import sqrt
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

if __name__ == '__main__':
    mainFunction()