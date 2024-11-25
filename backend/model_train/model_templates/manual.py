import json
import time
import boto3
import joblib
import argparse
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ml_def_and_train_module import def_and_train_ml

def main():
    
    try:
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
        columns_selected = metadata.get("columns_selected")
        columns_selected_for_train = metadata.get("columns_selected_for_train",[]) #

        # Load the dataset
        input_data_path = 'dataset.csv'  # Adjust if the data path differs
        df = pd.read_csv(input_data_path)
        if columns_selected:
            if "unit_number" in initial_columns:
                df = df[columns_selected_for_train + ["unit_number", "RUL"]]
            else:
                df = df[columns_selected_for_train + ["RUL"]]
        print("Dataset ready")

        # Fit the model
        model, X_test , y_test, model_type, best_params, architecture, time_steps = def_and_train_ml(df)
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
        if model_type == "sklearn":
            joblib.dump(model,"model.joblib")
            bucket_name = model_bucket
            s3_model_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".joblib"
            s3_client.upload_file("model.joblib",bucket_name,s3_model_key)
        elif model_type == "tensorflow":
            model.save("model.h5")
            bucket_name = model_bucket
            s3_model_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".h5"
            s3_client.upload_file("model.h5",bucket_name,s3_model_key)
        print(f"Model uploaded to s3://{bucket_name}/{s3_model_key}")

        columns_used_for_model = [col for col in df.columns if col not in ["RUL", "unit_number", "time_cycle_unit"]]
        
        # Create and upload model-metadata json file
        model_metadata = {}
        model_metadata["user_id"] = user_id
        model_metadata["actual_machine_id"] = metadata.get('actual_machine_id')
        model_metadata["unique_key"] = unique_key
        model_metadata["transformations_info"] = metadata.get('transformations_info')
        model_metadata["model_s3_key"] = s3_model_key
        model_metadata["time_steps"] = time_steps
        model_metadata["preprocessing_models_bucket"] = "preprocessing-models-bucket"
        model_metadata["Model"] = model_name
        model_metadata["R2-Score"]  = r2
        model_metadata["Mean Absolute Error"] = mae
        model_metadata["Mean-Squared-Error"] = mse
        model_metadata["model_type"] = model_type 
        model_metadata["Dataset-Version"] = dataset_version
        model_metadata["Hyper-Parameters"] = best_params
        model_metadata["Architecture"] = architecture
        model_metadata["Columns_used"] = columns_used_for_model
        model_metadata["model_id"] = args.model_id
        model_metadata["rul_units"] = metadata['rul_unit']
        model_metadata_json_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".json"
        s3_client.put_object(Bucket=model_bucket, Key=model_metadata_json_key, Body=json.dumps(model_metadata))
        print(f"Json(model-metadata) uploaded to bucket: {model_bucket}")

        architecture__hyper_parameters = {"hyper_parameters": best_params,"architecture": architecture}

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

    except Exception as e :
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