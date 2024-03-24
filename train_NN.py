import pandas as pd
import os
import zipfile
from itertools import combinations
import time
from tqdm import tqdm
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

root_dir = os.path.abspath(".")
data_dir = os.path.abspath("data")
root_dir, data_dir

# Check if a GPU is available and move the model and tensors to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_parquet_files_pandas(folder_path):
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.pq')]
    print(folder_path)
    dfs = []
    for file in sorted(parquet_files, key = lambda x: x.split("_")[2]):
        file_path = os.path.join(folder_path, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
        print("file = ",file_path)
        # break

    # Concatenate all DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    return result_df
X_train = read_parquet_files_pandas(os.path.join(data_dir, "Train_X"))
Y_train = y_train = read_parquet_files_pandas(os.path.join(data_dir, "Train_Y"))
X_val = read_parquet_files_pandas(os.path.join(data_dir, "Eval_X"))
Y_val = y_val = read_parquet_files_pandas(os.path.join(data_dir, "Eval_Y"))

X_train, X_test = train_test_split(X_train, train_size=0.1, random_state=42)
Y_train, Y_test = train_test_split(Y_train, train_size=0.1, random_state=42)

print(X_train.head())
print("X Train head")


X_train.describe()
print("X Train desc")


print(Y_train.describe())
print("Y train desc")


print()
print('---------------Training-----------------------------------')

# Assuming X_train and Y_train are your training data and labels
# Similarly, X_val is your validation data

# Convert your data to PyTorch tensors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# print(X_train.describe())
# print("X Train scaled desc")

# Move the data to the same device as the model
X_train_tensor = torch.Tensor(X_train).to(device)
X_test_tensor = torch.Tensor(X_test).to(device)
Y_train_tensor = torch.Tensor(Y_train.values).to(device)
X_val_tensor = torch.Tensor(X_val).to(device)
Y_val_tensor = torch.Tensor(Y_val.values).to(device)

# Create a custom dataset and data loaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the neural network model
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Adding a hidden layer with 64 neurons
        self.relu1 = nn.ReLU()  # Adding ReLU activation function
        self.fc2 = nn.Linear(64, 32)  # Adding another hidden layer with 32 neurons
        self.relu2 = nn.ReLU()  # Adding ReLU activation function
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = 14
model = RegressionNN(input_size).to(device)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()  # Use L1Loss for mean absolute error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15

for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_Y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        # print(f'Epoch [{epoch+1}/{num_epochs}], Step Loss: {loss.item():.4f}')


    # Print the loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the validation set
with torch.no_grad():
    model.eval()
    predictions = model(X_val_tensor)
    model.train()

# Print the predictions on the validation set
print("Predictions on the validation set:", predictions)

import numpy as np

def weighted_absolute_error(y_true, y_pred):
    """Calculates the weighted absolute error metric.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Weighted absolute error score.
    """
    errors = np.abs(y_true - y_pred)
    points = np.zeros(4)

    points[0] = np.count_nonzero(errors <= 0.05)
    points[1] = np.count_nonzero((errors > 0.05) & (errors <= 0.1))
    points[2] = np.count_nonzero((errors > 0.1) & (errors <= 0.5))

    points = points / len(y_true) * [1, 0.5, 0.25, 0]
    return sum(points)

def error_duration(y_true, y_pred):
    """Calculates the error duration metric.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Error duration score.
    """
    errors = np.abs(y_true - y_pred)
    error_blocks = (errors > 0.1).astype(int)
    error_block_durations = np.diff(np.where(error_blocks[:-1] != error_blocks[1:])[0]) + 1

    points = np.zeros(3)
    points[0] = np.count_nonzero(errors <= 0.1) / len(y_true)

    points[1] = np.count_nonzero((error_block_durations >= 2) & (error_block_durations <= 10)) / (
        len(y_true) / 8
    )
    points[2] = 1 - np.count_nonzero(error_block_durations > 10) / (len(y_true) / 2)

    points = points * [1, 0.25, 0]
    return sum(points)

def evaluate_regression_model(y_true, y_pred):
    """Evaluates a regression model using the weighted absolute error and error duration metrics.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: Dictionary containing the weighted absolute error and error duration scores.
    """
    scores = {}
    scores["weighted_absolute_error"] = weighted_absolute_error(y_true, y_pred)
    scores["error_duration"] = error_duration(y_true, y_pred)
    return scores


print("----------------Eval Metrics-------------------------------")
print(evaluate_regression_model(Y_val.to_numpy(),predictions.cpu().detach().numpy()))