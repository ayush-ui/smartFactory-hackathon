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

# Define the neural network model with Dropout and Batch Normalization
class RegressionNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ... (previous code)

# Initialize the model, loss function, and optimizer with weight decay
input_size = 14
model = RegressionNN(input_size).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adjust weight decay as needed
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Adjust parameters as needed

# Training loop with early stopping
num_epochs = 15
best_loss = float('inf')
patience = 3
current_patience = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}')

    # Adjust learning rate
    scheduler.step()

    # Evaluate the model on the validation set
    with torch.no_grad():
        model.eval()
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, Y_val_tensor)
        model.train()

    print(f'Validation Loss: {val_loss.item():.4f}')

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        current_patience = 0
    else:
        current_patience += 1
        if current_patience == patience:
            print("Early stopping. The model has stopped improving on the validation set.")
            break

# ... (rest of the code)


# Evaluate the model on the validation set
with torch.no_grad():
    model.eval()
    predictions = model(X_val_tensor)
    model.train()

# Print the predictions on the validation set
print("Predictions on the validation set:", predictions)

# Assume X_test is the test data tensor

with torch.no_grad():
    model.eval()
    test_predictions = model(torch.Tensor(X_test_tensor).to(device))
    model.train()

# Calculate accuracy on the test set
correct_predictions = (test_predictions.cpu().detach().numpy() == Y_test.values).sum()
accuracy = correct_predictions.item() / len(Y_test)
print("Accuracy on the test set:", accuracy)

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
