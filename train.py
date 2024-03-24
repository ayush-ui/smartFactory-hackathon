import os
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import  mean_absolute_error
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


root_dir = os.path.abspath(".")
data_dir = os.path.abspath("data")

def read_parquet_files_pandas(folder_path):
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.pq')]
    print(folder_path)
    dfs = []
    for file in sorted(parquet_files, key = lambda x: x.split("_")[2]):
        file_path = os.path.join(folder_path, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
        print("file = ",file_path)

    result_df = pd.concat(dfs, ignore_index=True)

    return result_df

def min_max_scale_data(df, scaler=None):
    if df.empty:
        return df

    if scaler is None:
        scaler = MinMaxScaler()

    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df_scaled, scaler


X_train = read_parquet_files_pandas(os.path.join(data_dir, "Train_X"))
Y_train = y_train = read_parquet_files_pandas(os.path.join(data_dir, "Train_Y"))
X_val = read_parquet_files_pandas(os.path.join(data_dir, "Eval_X"))
Y_val = y_val = read_parquet_files_pandas(os.path.join(data_dir, "Eval_Y"))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler() 


X_train_scaled, scaler_x = min_max_scale_data(X_train)
X_val_scaled = scaler_x.transform(X_val)

print(X_train.head())
print("X Train head")


X_train.describe()
print("X Train desc")


print(X_train_scaled.describe())
print("X Train scaled desc")

print(Y_train.describe())
print("Y train desc")


print()
print('---------------Training-----------------------------------')

t = time.time()

models = {
    # 'Linear Regression': {
    #     'model': LinearRegression(),
    #     'param_grid': {}
    # },
    # 'Gradient Boosting Regressor': {
    #     'model': GradientBoostingRegressor(),
    #     'param_grid': {
    #         'regression__n_estimators': [50, 100, 200],
    #         'regression__learning_rate': [0.01, 0.1, 0.2],
    #         'regression__max_depth': [3, 5, 7],
            
    #     }
    # },
    'Random Forest Regressor': {
        'model': RandomForestRegressor(),
        'param_grid': {
            'regression__n_estimators': [50, 100],
            'regression__max_depth': [None, 10, 20],
            
        }
    }
}

for model_name, model_info in models.items():
    print(f"Training {model_name}...")

    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regression', model_info['model'])
    ])

    param_grid = {
        'feature_selection__k': [5, 10, 14],
        **model_info['param_grid']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train.to_numpy().ravel())

    best_model = grid_search.best_estimator_

    y_pred_val = best_model.predict(X_val_scaled)
    mae_val = mean_absolute_error(y_val.to_numpy().ravel(), y_pred_val)

    print(f"Best {model_name} Model: {best_model}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Validation Mean Squared Error: {mae_val}")
    print("\n")
    print("Time = ", time.time()-t)
    

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

print("---Evaluation Metrics---")
print(evaluate_regression_model(Y_val.to_numpy().ravel(),y_pred_val))

if not os.path.exists("model"):
    os.mkdir("model")
    
import joblib
joblib.dump(best_model,"model/best_model.joblib")
joblib.dump(scaler_x,"model/scaler_x.joblib")