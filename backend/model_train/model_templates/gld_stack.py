import json
import time
import boto3
import argparse
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def modify_and_split_dataset(data):
    # Reshape data into sequences for LSTM (n_samples, time_steps, n_features)
    def create_sequences(data, time_steps = 10, target_col = "RUL"):
      sequences = []
      targets = []

      # if unit number is not available generate it using rul
      if 'unit_number' not in data.columns:
        # Creating a boolean Series that is True where `RUL` is zero, indicating the end of a unit
        end_of_unit = data['RUL'] == 0

        # Use cumulative sum to assign unique unit numbers.
        # This increments the unit number every time `RUL == 0` is encountered.
        data['unit_number'] = end_of_unit.cumsum() + 1

      feature_columns = [col for col in data.columns if col not in ['unit_number', 'time_cycle_unit', target_col]]

      for unit_number in data['unit_number'].unique(): #work using rul not unit number or generate unit number if not present
          machine_data = data[data['unit_number'] == unit_number]
          for i in range(len(machine_data) - time_steps):
              seq = machine_data[feature_columns].iloc[i:i + time_steps].values
            #   seq = machine_data.iloc[i:i + time_steps].values
              label = machine_data.iloc[i + time_steps][target_col]
              sequences.append(seq)
              targets.append(label)

      return np.array(sequences), np.array(targets)

    X_seq, y_seq = create_sequences(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_lstm_layers, num_gru_layers, num_dense_layers):
    model = Sequential()
    architecture = {"type": "Sequential","layers": []}
    
    # Add LSTM layers
    for _ in range(num_lstm_layers):
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        architecture["layers"].append({"type": "LSTM", "units": 128, "return_sequences": True})
        architecture["layers"].append({"type": "BatchNormalization"})
    
    # Add GRU layers
    for i in range(num_gru_layers):
        return_sequences = i < num_gru_layers - 1  # False only for the last GRU layer
        model.add(GRU(64, return_sequences=return_sequences))
        model.add(BatchNormalization())
        architecture["layers"].append({"type": "GRU", "units": 64, "return_sequences": return_sequences})
        architecture["layers"].append({"type": "BatchNormalization"})
    
    # Add Dense layers
    for _ in range(num_dense_layers):
        model.add(Dense(32, activation='relu'))
        architecture["layers"].append({"type": "Dense", "units": 32, "activation": "relu"})
    
    # Output layer
    model.add(Dense(1, activation='linear'))
    architecture["layers"].append({"type": "Dense", "units": 1, "activation": "linear"})
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    architecture["optimizer"] = "Adam"
    architecture["loss"] = "mse"
    architecture["learning_rate"] = 0.001

    return model, architecture

def compare_models(X_train, y_train, X_test, y_test, input_shape, best_model, best_arch, lstm_range=(1, 3), gru_range=(1, 3), dense_range=(1, 3)):
    results = []

    prev_r2 = 0

    for num_lstm_layers in range(lstm_range[0], lstm_range[1] + 1):
        for num_gru_layers in range(gru_range[0], gru_range[1] + 1):
            for num_dense_layers in range(dense_range[0], dense_range[1] + 1):
                
                # Build model
                model, arch = build_model(input_shape, num_lstm_layers, num_gru_layers, num_dense_layers)
                
                # Model training
                model.fit(X_train, y_train, epochs=75, batch_size=64, validation_data=(X_test, y_test), verbose=0)
                
                # Predictions and evaluation
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                if r2 > prev_r2:
                    best_model = model
                    best_arch = arch
                    prev_r2 = r2
                
                # Store results
                results.append({
                    "num_lstm_layers": int(num_lstm_layers),
                    "num_gru_layers": int(num_gru_layers),
                    "num_dense_layers": int(num_dense_layers),
                    "MSE": mse,
                    "MAE": mae,
                    "R2": r2
                })

                print(f"LSTM: {num_lstm_layers}, GRU: {num_gru_layers}, Dense: {num_dense_layers} | MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    return pd.DataFrame(results), best_model, best_arch

def train_gld_stack(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], X_train.shape[2])
    best_model = None
    best_arch = None
    results_df , model, architecture = compare_models(X_train, y_train, X_test, y_test, input_shape, best_model, best_arch)

    print(results_df)
    # best_row = results_df.nlargest(1,"R2")
    # print(best_row)
    
    architecture = {"model" : architecture}

    return model, architecture

def main():
    
    try :
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Train ML Model .")
        parser.add_argument('json_bucket_name', type=str, help="Name of the bucket where json file exists .")
        parser.add_argument('json_key', type=str, help="Name of the json file .")
        parser.add_argument('model_id', type=str, help="MODEL ID .")
        args = parser.parse_args()
        print("Arguments parsed :")
        print(f"json_bucket_name = {args.json_bucket_name}")
        print(f"json_key = {args.json_key}")
        print(f"model_id = {args.model_id}")

        # Download and extract json contents
        s3_client = boto3.client('s3')
        json_file_obj = s3_client.get_object(Bucket=args.json_bucket_name, Key=args.json_key)
        json_file_content = json_file_obj['Body'].read().decode('utf-8')
        metadata = json.loads(json_file_content)
        user_id = metadata.get("user_id")
        timestamp = str(time.time())
        unique_key = metadata.get("unique_key")
        model_name = metadata.get("model_selected")
        model_bucket = metadata.get("model_bucket")
        dataset_version = metadata.get("dataset_version")
        model_selected_by = metadata.get("model_selected_by")
        initial_columns = metadata.get("final_columns")
        columns_selected =  metadata.get("columns_selected")
        columns_selected_for_train = metadata.get("columns_selected_for_train",[]) #

        # Load the dataset
        input_data_path = 'dataset.csv'# Adjust if the data path differs
        df = pd.read_csv(input_data_path)
        if columns_selected:
            if "unit_number" in initial_columns:
                df = df[columns_selected_for_train + ["unit_number", "RUL"]]
            else:
                df = df[columns_selected_for_train + ["RUL"]]
        print("Dataset ready")

        X_train, X_test, y_train, y_test = modify_and_split_dataset(df)
        print("Dataset split completed")

        # Fit the model
        model , architecture = train_gld_stack(X_train, y_train, X_test, y_test)
        print("Model trained.")

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R^2 Score: {r2:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")

        # Save the model
        model.save("model.h5")
        bucket_name = model_bucket
        s3_model_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".h5"
        s3_client.upload_file("model.h5",bucket_name,s3_model_key)
        print(f"Model uploaded to s3://{bucket_name}/{s3_model_key}")

        columns_used_for_model = [col for col in df.columns if col not in ["RUL", "unit_number", 'time_cycle_unit']]
        # Create and upload model-metadata json file
        model_metadata = {}
        model_metadata["user_id"] = user_id
        model_metadata["actual_machine_id"] = metadata.get('actual_machine_id')
        model_metadata["unique_key"] = unique_key
        model_metadata["transformations_info"] = metadata.get('transformations_info')
        model_metadata["model_s3_key"] = s3_model_key
        model_metadata["time_steps"] = 10
        model_metadata["preprocessing_models_bucket"] = "preprocessing-models-bucket"
        model_metadata["Model"] = model_name
        model_metadata["R2-Score"]  = r2
        model_metadata["Mean Absolute Error"] = mae
        model_metadata["Mean-Squared-Error"] = mse
        model_metadata["model_type"] = "tensorflow" 
        model_metadata["Dataset-Version"] = dataset_version
        model_metadata["Architecture"] = architecture
        model_metadata["Columns_used"] = columns_used_for_model
        model_metadata["model_id"] = args.model_id
        model_metadata["rul_units"] = metadata['rul_unit']
        model_metadata_json_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".json"
        s3_client.put_object(Bucket=model_bucket, Key=model_metadata_json_key, Body=json.dumps(model_metadata))
        print(f"Json(model-metadata) uploaded to bucket: {model_bucket}")

        architecture__hyper_parameters = architecture

        # Update status variable
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': unique_key,
            'model_id': args.model_id,
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'architecture/hyper-parameters': architecture__hyper_parameters,
            'columns_used': columns_used_for_model,
            'model_metadata_s3_key': model_metadata_json_key,
            'model_key': s3_model_key,
            'operation': 'update',
            'object': 'model',
            'status': 'Model training is completed .'
        }
        requests.post(firebase_update_url, json=payload)
    
    except Exception as e:
        print(f"Error occured while trying to train the model : {str(e)}")
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': unique_key,
            'model_id': args.model_id,
            'operation': 'update',
            'object': 'model',
            'status': f'Error in Model training : {str(e)} .'
        }
        requests.post(firebase_update_url, json=payload)

if __name__=="__main__":
    main()