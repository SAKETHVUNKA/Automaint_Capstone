import json
import boto3
import joblib
import argparse
import requests
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, r_regression

def load_data(file_path, has_headers=True):
    data = pd.read_csv(file_path)
    return data

def select_top_n_columns(df, has_unit_number, has_time_cycle_unit, target_column="RUL", n=25, score_func_name='mutual_info_regression'):
    """
    Select the top n columns from a dataframe based on a selected scoring function for regression tasks.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The column name of the target variable.
        n (int): The number of top features to select. Default is 20.
        score_func_name (str): The scoring function to use. Should be one of
                               'f_regression', 'mutual_info_regression', or 'r_regression'.
        has_unit_number (bool): Whether to remove and later add back the 'unit_number' column.

    Returns:
        pd.DataFrame: DataFrame with the top n selected features.
    """

    # Map of available scoring functions for regression tasks
    score_funcs = {
        'f_regression': f_regression,
        'mutual_info_regression': mutual_info_regression,
        'r_regression': r_regression
    }

    # Remove unwanted columns based on flags
    remove_columns = [target_column]
    if has_unit_number:
        remove_columns.append('unit_number')
    if has_time_cycle_unit:
        remove_columns.append('time_cycle_unit')

    # Separate features and target
    X = df.drop(columns=remove_columns).copy()
    y = df[target_column].copy()

    if X.shape[1] < n:
        return df

    # Check if the given score_func_name is valid
    if score_func_name not in score_funcs:
        raise ValueError(f"Invalid score_func_name: {score_func_name}. Choose from 'f_regression', 'mutual_info_regression', or 'r_regression'.")

    # Select the appropriate scoring function
    score_func = score_funcs[score_func_name]

    # Select the top n features using SelectKBest
    selector = SelectKBest(score_func=score_func, k=n)
    X_new = selector.fit_transform(X, y)

    # Get the selected feature names
    selected_columns = X.columns[selector.get_support(indices=True)]
    selected_columns = selected_columns.tolist()

    # Rebuild DataFrame with selected features and the target
    result_df = pd.DataFrame(np.column_stack([X_new, y]), columns=list(selected_columns) + [target_column])

    # Add back the 'unit_number' column if it was in the original dataset
    if has_unit_number:
        result_df['unit_number'] = df['unit_number'].values
    if has_time_cycle_unit:
        result_df['time_cycle_unit'] = df['time_cycle_unit'].values

    return result_df,selected_columns

def calculate_first_order_derivative(series):
    return series.diff().fillna(0)

def calculate_second_order_derivative(series):
    return series.diff().diff().fillna(0)

def generate_lagged_features(df, lags=[1, 2, 3, 4, 5]):
    lagged_features = pd.DataFrame(index=df.index)
    for lag in lags:
        for col in df.columns:
            lagged_features[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(0)
    return lagged_features

def apply_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(df)
    return pd.Series(clusters, name='KMeans_Cluster', index=df.index),kmeans

def apply_anomaly_detection(df):#generate anomaly score for individual columns instead of entire row
    iso_forest = IsolationForest(contamination=0.1, random_state=0)
    anomaly_scores = iso_forest.fit_predict(df)
    return pd.Series(anomaly_scores, name='Anomaly_Score', index=df.index),iso_forest

def generate_cross_time_series_features(df):
    cross_features = pd.DataFrame(index=df.index)
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i + 1:]:
            cross_features[f'{col1}_x_{col2}'] = (df[col1] * df[col2]).fillna(0)
    return cross_features

def process_sensor_data(df, has_unit_number, has_time_cycle_unit, auto_corr_lag=10):
    columns_to_remove = ['RUL']
    if has_unit_number:
        columns_to_remove.append('unit_number')
    if has_time_cycle_unit:
        columns_to_remove.append('time_cycle_unit')

    # Store original columns
    original_columns = df.columns.tolist()

    # Store columns not to be removed
    columns_to_include = [col for col in original_columns if col not in columns_to_remove]

    # Remove specific columns for feature generation
    df_to_process = df.drop(columns=columns_to_remove)

    # Store removed columns to re-add later
    removed_columns = df[columns_to_remove].copy()

    # Generate lagged features
    lagged_features = generate_lagged_features(df_to_process, lags=list(range(1, 6)))

    # Calculate first-order derivatives
    first_order_derivatives = df_to_process.apply(calculate_first_order_derivative)
    first_order_derivatives.columns = [f'FirstOrder_{col}' for col in df_to_process.columns]

    # Calculate second-order derivatives
    second_order_derivatives = df_to_process.apply(calculate_second_order_derivative)
    second_order_derivatives.columns = [f'SecondOrder_{col}' for col in df_to_process.columns]

    # Apply KMeans clustering
    kmeans_clusters,kmeans_model = apply_kmeans(df_to_process)

    # Apply anomaly detection
    anomaly_scores,anomaly_model = apply_anomaly_detection(df_to_process)

    # Generate cross-time series features
    cross_features = generate_cross_time_series_features(df_to_process)

    # Combine all features
    all_features = pd.concat([lagged_features, first_order_derivatives, second_order_derivatives, kmeans_clusters, anomaly_scores, cross_features], axis=1)

    # Re-add removed columns in specified positions with capitalized names
    if 'time_cycle_unit' in removed_columns.columns:
        all_features.insert(0, 'time_cycle_unit', removed_columns['time_cycle_unit'])
    if 'unit_number' in removed_columns.columns:
        all_features.insert(0, 'unit_number', removed_columns['unit_number'])
    if 'RUL' in removed_columns.columns:
        all_features['RUL'] = removed_columns['RUL']

    # Re-add original columns from df_to_process
    for col in columns_to_include:
        if col not in all_features.columns:
            all_features[col] = df[col]

    return all_features,kmeans_model,anomaly_model

def main():
    print("in main")
    try: 
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Process sensor data and apply feature selection, KMeans, and Anomaly Detection.")
        parser.add_argument('unique_key', type=str, help="unique key for the dataset")
        parser.add_argument('in_json_bucket_name', type=str, help="Name of the S3 bucket to DOWNLOAD JSON_2")
        parser.add_argument('in_json_key', type=str, help="Name of the JSON_2")
        args = parser.parse_args()
        print("Arguments parsed.")
        print("Unique Key :",args.unique_key)
        print("Json Bucket :",args.in_json_bucket_name)
        print("Json Key :",args.in_json_key)

        # Download json_2
        s3_client = boto3.client('s3')
        json_file_obj = s3_client.get_object(Bucket=args.in_json_bucket_name, Key=args.in_json_key)
        json_file_content = json_file_obj['Body'].read().decode('utf-8')
        metadata = json.loads(json_file_content)
        user_id = metadata['user_id']
        has_unit_number = metadata.get("has_unit_number")
        has_time_cycle_unit = metadata.get("has_time_cycle_unit")
        final_dataset_bucket = metadata.get("final_dataset_bucket")
        preprocessing_models_bucket = metadata.get("preprocessing_models_bucket")

        # Load and process the dataset
        input_data_path = "dataset.csv"
        df = load_data(input_data_path)
        print("Dataset loaded .")
        processed_data, kmeans_model, anomaly_model = process_sensor_data(df, has_unit_number, has_time_cycle_unit)
        print("Feature generation done .")

        # Apply feature selection
        final_data, selected_columns = select_top_n_columns(processed_data,has_unit_number, has_time_cycle_unit)
        print("Feature selection done")

        # Upload the final dataset to the specified S3 bucket
        csv_buffer = StringIO()
        final_data.to_csv(csv_buffer, index=False)
        output_key = args.unique_key + "_final_dataset.csv"  # Set the S3 object key based on the file name
        s3_client.put_object(Bucket=final_dataset_bucket, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Final Dataset uploaded to bucket: {final_dataset_bucket}")

        # Save and upload the pre-processing models to the specified S3 bucket
        if "KMeans_Cluster" in selected_columns:
            kmeans_key = f"{args.unique_key}_kmeans_model.joblib"
            joblib.dump(kmeans_model, kmeans_key)
            s3_client.upload_file(kmeans_key, preprocessing_models_bucket, kmeans_key)
            print(f"Pre-Processing Model (Kmeans-Clustering) uploaded to bucket: {preprocessing_models_bucket}")
        if "Anomaly_Score" in selected_columns:
            anomaly_key = f"{args.unique_key}_anomaly_model.joblib"
            joblib.dump(anomaly_model, anomaly_key)
            s3_client.upload_file(anomaly_key, preprocessing_models_bucket, anomaly_key)
            print(f"Pre-Processing Model (Anomaly-Scores) uploaded to bucket: {preprocessing_models_bucket}")

        # Modify json_2 and upload it as json_3
        metadata1 = {}
        metadata1['final_dataset_name'] = args.unique_key + "_final_dataset.csv"
        metadata1["model_selected_by"] = "automatic"
        metadata1["selected_columns"] = selected_columns
        metadata1["columns_selected"] = False
        metadata1["model_selected"] = "lstm"
        metadata.update(metadata1)
        json_key = f"{args.unique_key}_3.json"
        s3_client.put_object(Bucket=final_dataset_bucket, Key=json_key, Body=json.dumps(metadata))
        print(f"Json_3 uploaded to bucket: {final_dataset_bucket}")

        # Update status variable
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': args.unique_key,
            'operation': 'update',
            'object': 'machine',
            'status': 'Pre-processing is completed .'
        }
        requests.post(firebase_update_url, json=payload)
    
    except Exception as e:

        # Update status variable
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': args.unique_key,
            'operation': 'update',
            'object': 'machine',
            'status': 'Error in Pre-processing .'
        }
        requests.post(firebase_update_url, json=payload)

        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()