# Import necessary libraries
import pickle as pkl
import os
import joblib
import pandas as pd
import warnings
import sys

def process_test_data(test_folder_path):
    # If test_folder_path is not provided as a command line argument, set it to the "input" folder in the script directory
    if not test_folder_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_folder_path = os.path.join(script_dir, "Input")

        # If the "input" folder does not exist, raise an error
        if not os.path.exists(test_folder_path):
            raise FileNotFoundError("Either provide the test folder path as a command line argument or create an 'Input' folder in the script directory")

    # Filter out a specific warning by category (ignore sklearn warnings)
    warnings.filterwarnings("ignore", module="sklearn")

    # Define the path to the directory containing the pre-trained models
    models_path = os.path.join(os.path.abspath("."), "model")

    # Define the names of the pre-trained model and scaler
    # model_name = "model_RFR.joblib"
    # scaler_name = "scaler_x_RFR.joblib"
    
    model_name = "best_model.joblib"
    scaler_name = "scaler_x.joblib"

    # model_name = "best_model.pkl"
    # scaler_name = "scaler_x.pkl"

    # Construct the full paths for the pre-trained model and scaler
    model_path = os.path.join(models_path, model_name)
    scaler_path = os.path.join(models_path, scaler_name)

    # Load the pre-trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # with open(os.path.join(models_path,model_name), "rb") as f:
    #     model = pkl.load(f)

    # with open(os.path.join(models_path,scaler_name), "rb") as f:
    #     scaler = pkl.load(f)
    
    # Iterate over each file in the specified test data folder
    for file_name in filter(lambda x: x.endswith(".pq"), os.listdir(test_folder_path)):
        # Construct the full file path
        file_path = os.path.join(test_folder_path, file_name)

        # Read the Parquet file into a Pandas DataFrame
        df = pd.read_parquet(file_path)
        # df = df.iloc[:,model.named_steps['feature_selection'].get_support()]

        # Transform the input data using the pre-defined scaler
        x_transform = scaler.transform(df)

        # Make predictions using the pre-trained model
        predictions = model.predict(x_transform)

        # Create a new DataFrame with the predictions and set datetime index
        output_df = pd.DataFrame(predictions, columns=['predictions'])
        output_df.index = pd.to_datetime(df.index)

        # Generate the output file name by adding "_pred" to the original file name
        output_filename = file_name.split('.')[0] + "_pred.pq"

        # Save the predictions DataFrame to a new Parquet file
        output_df.to_parquet(output_filename)

        # Print a message indicating that the file has been saved
        print(f"{output_filename} saved")

# Check if the script is executed directly
if __name__ == "__main__":
    # The first command line argument is at index 1
    test_folder_path = sys.argv[1] if len(sys.argv) == 2 else None
    process_test_data(test_folder_path)
