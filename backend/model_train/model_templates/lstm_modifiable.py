import json
import time
import boto3
import argparse
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
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
        data['unit_number'] = 1  # Initialize with 0 or another default value 

        unit_counter = 1  # Start with unit number 1
        for i in range(len(data)):
            # Assign the unit number only to the row after RUL == 0
            data['unit_number'].loc[i] = unit_counter
            # If the current RUL is 0, increment the unit number
            if data['RUL'].iloc[i] == 0 and i + 1 < len(data):  # Ensure there's a next row
                unit_counter += 1

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

def build_lstm_model(input_shape, num_lstm_layers, num_dense_layers, use_batch_normalization, use_dropout, dropout_rate):
    architecture = {"type": "Sequential","layers": []}
    model = Sequential()

    for i in range(num_lstm_layers):
        return_sequences_val = (i < num_lstm_layers - 1) 
        if i==0:
            model.add(LSTM(64, return_sequences=return_sequences_val, input_shape=input_shape))
            architecture["layers"].append({"type": "LSTM", "units": 64, "return_sequences": return_sequences_val, "input_shape": input_shape})
        else:
            model.add(LSTM(64, return_sequences=return_sequences_val))
            architecture["layers"].append({"type": "LSTM", "units": 64, "return_sequences": return_sequences_val})
        if use_dropout:
            architecture["layers"].append({"type": "Dropout","dropout_value": dropout_rate})
            model.add(Dropout(dropout_rate))
    if use_batch_normalization:
        model.add(BatchNormalization())
        architecture["layers"].append({"type": "BatchNormalization"})

    for i in range(num_dense_layers):    
        model.add(Dense(64, activation='relu'))
        architecture["layers"].append({"type": "Dense", "units": 64, "activation": "relu"})
    
    model.add(Dense(1, activation='linear'))
    architecture["layers"].append({"type": "Dense", "units": 1, "activation": "linear"})

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    architecture = {"model" : architecture}

    return model, architecture

def train_lstm(X_train, y_train, X_test, y_test, num_lstm_layers, num_dense_layers, use_batch_normalization, use_dropout, dropout_rate):
    model, architecture = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_lstm_layers, num_dense_layers, use_batch_normalization, use_dropout, dropout_rate)

    # Fit the model
    history = model.fit(X_train, y_train, epochs=65, batch_size=64, validation_data=(X_test, y_test), verbose=1)

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
        columns_selected_for_train = metadata.get("columns_selected_for_train",[]) 
        use_batch_normalization = metadata.get("batch_normalization", False) 
        use_dropout = metadata.get("dropout", False) 
        dropout_rate = metadata.get("dropout_rate", 0.2)
        num_lstm_layers = metadata.get("num_lstm_layers", 1)
        num_dense_layers = metadata.get("num_dense_layers", 1)

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
        model , architecture = train_lstm(X_train, y_train, X_test, y_test, num_lstm_layers, num_dense_layers, use_batch_normalization, use_dropout, dropout_rate)
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