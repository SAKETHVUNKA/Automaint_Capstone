import json
import time
import boto3
import joblib
import optuna
import argparse
import requests
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_xgb(X_train, y_train, n_trials=50):
    # Define the objective function
    def objective_xgb(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        # max_depth = trial.suggest_int('max_depth', 3, 20)
        # min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        # subsample = trial.suggest_float('subsample', 0.5, 1.0)
        # colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        
        # Create the XGBRegressor model with suggested hyperparameters
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            # max_depth=max_depth,
            # min_child_weight=min_child_weight,
            # subsample=subsample,
            # colsample_bytree=colsample_bytree,
            gamma=gamma,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        # Evaluate the model using cross-validation
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1).mean()
        return score

    # Create a study for minimizing the objective function
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials)

    # Get the best hyperparameters
    best_params = study_xgb.best_params

    # Train the final model with the best hyperparameters
    best_model = XGBRegressor(**best_params, objective='reg:squarederror', n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params

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
        columns_selected = metadata.get("columns_selected")
        columns_selected_for_train = metadata.get("columns_selected_for_train",[]) #

        # Load the dataset
        input_data_path = 'dataset.csv'  # Adjust if the data path differs
        df = pd.read_csv(input_data_path)
        print("Dataset ready")

        # Split the dataset into features and target variable (and select only the columns selected by user) 
        X = df.drop('RUL', axis=1)
        X = X.drop('unit_number', axis=1, errors='ignore')
        X = X.drop('time_cycle_unit', axis=1, errors='ignore')
        if columns_selected :
            X = X[columns_selected_for_train]
        y = df['RUL']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Dataset split completed")

        # Fit the model
        model , best_params = train_xgb(X_train,y_train)
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
        joblib.dump(model,"model.joblib")
        bucket_name = model_bucket
        s3_model_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".joblib"
        s3_client.upload_file("model.joblib",bucket_name,s3_model_key)
        print(f"Model uploaded to s3://{bucket_name}/{s3_model_key}")

        columns_used_for_model = X.columns.tolist()

        # Create and upload model-metadata json file
        model_metadata = {}
        model_metadata["user_id"] = user_id
        model_metadata["actual_machine_id"] = metadata.get('actual_machine_id')
        model_metadata["unique_key"] = unique_key
        model_metadata["transformations_info"] = metadata.get('transformations_info')
        model_metadata["model_s3_key"] = s3_model_key
        model_metadata["time_steps"] = 1
        model_metadata["preprocessing_models_bucket"] = "preprocessing-models-bucket"
        model_metadata["Model"] = model_name
        model_metadata["R2-Score"]  = r2
        model_metadata["Mean Absolute Error"] = mae
        model_metadata["Mean-Squared-Error"] = mse
        model_metadata["model_type"] = "sklearn"
        model_metadata["Dataset-Version"] = dataset_version
        model_metadata["Hyper-Parameters"] = best_params
        model_metadata["Columns_used"] = columns_used_for_model
        model_metadata["model_id"] = args.model_id
        model_metadata["rul_units"] = metadata['rul_unit']
        model_metadata_json_key = unique_key +"_"+ model_name +"_"+ model_selected_by +"_"+ timestamp + ".json"
        s3_client.put_object(Bucket=model_bucket, Key=model_metadata_json_key, Body=json.dumps(model_metadata))
        print(f"Json(model-metadata) uploaded to bucket: {model_bucket}")

        architecture__hyper_parameters = best_params

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