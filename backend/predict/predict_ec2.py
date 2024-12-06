import math
import json
import boto3
import joblib
import datetime
import argparse
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from tensorflow.keras.models import load_model

# Initialize a deque to store the last 10 received values
recent_values = pd.DataFrame()

def load_model_from_s3(bucket_name, model_key, type):
    """Helper function to load a model from S3."""
    s3 = boto3.client('s3')
    model_name = f'/tmp/{type}_model.joblib'
    s3.download_file(bucket_name, model_key, model_name)
    return joblib.load(model_name)

# Function to apply scaling transformations based on the transformation info
def apply_transformations(data, transformations_info):
    transformed_data = pd.DataFrame()
    transformations = transformations_info["transformations"]

    for feature, params in transformations.items():
        if feature in data.columns:
            if params["type"] == "scaling":
                mean = params["mean"]
                scale = params["scale"]
                transformed_data[feature] = (data[feature] - mean) / scale
            elif params["type"] == "encoding":
                # Retrieve encoding dictionary from transformations
                encoding_map = params["conversion"]
                transformed_data[feature] = data[feature].map(encoding_map)

    return transformed_data

def transform_to_selected_columns(df, selected_columns, kmeans_model, iso_forest_model):
    # Prepare an empty DataFrame to store the selected transformations
    transformed_df = pd.DataFrame(index=df.index)
    
    # Step 1: Generate only the required lagged features
    lagged_features = pd.DataFrame(index=df.index)
    for col in df.columns:
        for lag in range(1, 6):  # Up to lag 5 based on the example
            lagged_col_name = f'{col}_lag_{lag}'
            if lagged_col_name in selected_columns:
                lagged_features[lagged_col_name] = df[col].shift(lag).fillna(0)
    transformed_df = pd.concat([transformed_df, lagged_features], axis=1)
    
    # Step 2: Apply KMeans clustering only if required
    if 'KMeans_Cluster' in selected_columns:
        kmeans_cluster = pd.Series(kmeans_model.predict(df), name='KMeans_Cluster', index=df.index)
        transformed_df['KMeans_Cluster'] = kmeans_cluster
    
    # Step 3: Generate only the required cross features
    cross_features = pd.DataFrame(index=df.index)
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i + 1:]:
            cross_col_name = f'{col1}_x_{col2}'
            if cross_col_name in selected_columns:
                cross_features[cross_col_name] = (df[col1] * df[col2]).fillna(0)
    transformed_df = pd.concat([transformed_df, cross_features], axis=1)
    
    # Step 4: Apply anomaly detection only if required
    if 'Anomaly_Score' in selected_columns:
        anomaly_scores = pd.Series(iso_forest_model.predict(df), name='Anomaly_Score', index=df.index)
        transformed_df['Anomaly_Score'] = anomaly_scores
    
    # Step 5: Add required base features from df to match selected_columns
    available_base_features = [col for col in selected_columns if col in df.columns]
    base_features = df[available_base_features].copy()
    transformed_df = pd.concat([transformed_df, base_features], axis=1)
    
    # Step 6: Filter to selected columns
    final_df = transformed_df[selected_columns].copy()
    
    final_df = final_df.reindex(columns=selected_columns)

    return final_df

# Create a function to preprocess the received data
def preprocess_data(df, transformations_info, columns_used, kmeans_model, iso_forest_model):
    # Apply transformations to each record in recent_values
    scaled_df = apply_transformations(df, transformations_info)

    transformed_df = transform_to_selected_columns(scaled_df, columns_used, kmeans_model, iso_forest_model)
    
    return transformed_df

def func1(user_data):

    # Callback function for MQTT messages
    def on_message(client, userdata, message):
        global recent_values

        # Parse the incoming message
        data = json.loads(message.payload.decode('utf-8'))
        print(f"Received: {data}")

        if not any(value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))) for value in data.values()):
            
            # Check if recent_values is empty to initialize with columns
            if recent_values.empty:
                # Initialize the DataFrame with the first row as column names
                recent_values = pd.DataFrame(columns=data.keys())

            # If the DataFrame has reached the maximum number of rows, drop the oldest row
            if len(recent_values) >= user_data["time_steps"] + 5 :
                recent_values = recent_values.iloc[1:]  # Remove the topmost row

            # Append the new data as a new row
            recent_values = pd.concat([recent_values, pd.DataFrame([data])], ignore_index=True)

            # Check if we have enough values for prediction
            if len(recent_values) == user_data["time_steps"] + 5 :
                # Preprocess the data
                processed_data = preprocess_data(recent_values, user_data["transformations_info"], user_data["columns_used"], user_data['kmeans_model'], user_data['iso_forest_model'])

                if user_data["time_steps"] > 1:
                    input_sequence = processed_data.iloc[-user_data["time_steps"]:].values
                    input_sequence = input_sequence.reshape((1, user_data["time_steps"], input_sequence.shape[1]))
                elif user_data["time_steps"] == 1:
                    input_sequence = processed_data.iloc[[user_data["time_steps"]+4]]
                # Prediction
                model = user_data['predict_model']
                prediction = model.predict(input_sequence)
                prediction = prediction[0]
                if user_data['extension'] == "h5":
                    prediction = prediction[0]
                prediction = int(prediction)
                if prediction < 0:
                    prediction = 0
                print(f"Prediction: {prediction}")
                rul_unit = user_data["rul_unit"]
                prediction = str(prediction)+" "+rul_unit 

                new_status = f"Rul Prediction updated at {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}"
                update_status(user_data["user_id"], user_data["actual_machine_id"], prediction, new_status)

                #if possible clean the prediction like no negatives or better tell before than later

    return on_message

def update_status(user_id, machine_id, rul, new_status):
    firebase_update_url = 'https://kbkt311nic.execute-api.ap-south-1.amazonaws.com/default/firebase-update-automaint'
    payload = {
            'user_id': user_id,
            'machine_id': machine_id,
            'operation': 'update',
            'object': 'actual_machine',
            'rul': rul,
            'status': new_status
        }
    requests.post(firebase_update_url, json=payload)

def main():
    try: 
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Pre Process and predict RUL from live sensor data stream")
        parser.add_argument('model_json_key', type=str, help="unique key for the model's metadata json file")
        parser.add_argument('mqtt_topic', type=str, help="mqtt topic to which data will be streamed")
        args = parser.parse_args()
        print("Arguments parsed.")
        print("Model Json Key :",args.model_json_key)
        print("Mqtt Topic :",args.mqtt_topic)

        # Download json_2
        s3_client = boto3.client('s3')
        json_file_obj = s3_client.get_object(Bucket="ml-models-trained-automaint", Key=args.model_json_key)
        json_file_content = json_file_obj['Body'].read().decode('utf-8')
        metadata = json.loads(json_file_content)
        user_id = metadata.get('user_id')
        actual_machine_id = metadata.get('actual_machine_id')
        unique_key = metadata.get('unique_key') 
        transformations_info = metadata.get("transformations_info") 
        columns_used = metadata.get("Columns_used") 
        time_steps = metadata.get("time_steps") 
        model_id = metadata.get("model_id")
        preprocessing_models_bucket = metadata.get("preprocessing_models_bucket")
        rul_unit = metadata.get("rul_units","NA") 
        model_name = metadata.get("model_s3_key") 
        extension = model_name.split(".")[-1]
        local_model_file_name = f"model.{extension}"

        s3_client.download_file("ml-models-trained-automaint", model_name, local_model_file_name)
        if extension == "h5":
            model = load_model(local_model_file_name)
        elif extension == "joblib":
            model = joblib.load(local_model_file_name)

        kmeans_model = model
        iso_forest_model = model
        if 'KMeans_Cluster' in columns_used:
            # Load the KMeans model from S3
            kmeans_key = f"{unique_key}_kmeans_model.joblib"
            print(kmeans_key)
            kmeans_model = load_model_from_s3(preprocessing_models_bucket, kmeans_key, "kmeans")
            print("kmeans model downloaded")

        if 'Anomaly_Score' in columns_used:
            # Load the IsolationForest model from S3
            iso_forest_key = f"{unique_key}_anomaly_model.joblib"
            print(iso_forest_key)
            iso_forest_model = load_model_from_s3(preprocessing_models_bucket, iso_forest_key, "isolation_forest")
            print("isolation-forest model downloaded")

        # MQTT connection details
        CLIENT_ID = f"{user_id}_{model_id}" 
        ENDPOINT = "a32052tf0w5nxl-ats.iot.ap-south-1.amazonaws.com"
        TOPIC = args.mqtt_topic
        CERT_PATH = "./certificate_iot_core.pem.crt"
        KEY_PATH = "./private_key_iot_core.pem.key"
        ROOT_CA_PATH = "./root_ca_iot_core.pem"

        # Initialize MQTT client
        mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        mqtt_client.configureEndpoint(ENDPOINT, 8883)
        mqtt_client.configureCredentials(ROOT_CA_PATH, KEY_PATH, CERT_PATH)
        mqtt_client.configureOfflinePublishQueueing(-1)
        mqtt_client.configureDrainingFrequency(2) 
        mqtt_client.configureConnectDisconnectTimeout(10)
        mqtt_client.configureMQTTOperationTimeout(5)

        user_data = {
            "transformations_info" : transformations_info,
            "time_steps" : time_steps,
            "columns_used" : columns_used,
            "actual_machine_id" : actual_machine_id,
            "user_id" : user_id,
            "kmeans_model" : kmeans_model,
            "iso_forest_model" : iso_forest_model,
            "predict_model": model,
            "extension": extension,
            "rul_unit": rul_unit
        }

        # Connect to MQTT broker
        mqtt_client.connect()
        mqtt_client.subscribe(TOPIC, 1, func1(user_data))

        # Keep the script running to listen for messages
        update_status(user_id, actual_machine_id, "NA", "Predicting")

        firebase_update_url = 'https://kbkt311nic.execute-api.ap-south-1.amazonaws.com/default/firebase-update-automaint'
        payload = {
                'user_id': user_id,
                'machine_id': unique_key,
                'model_id': model_id,
                'operation': 'update',
                'object': 'model',
                'status': 'Predicting',
                'type': "only_status"
        }
        requests.post(firebase_update_url, json=payload)
        
        while True:
            pass

    except Exception as e :
        print(f"Error during prediction : {str(e)}")
        update_status(user_id, actual_machine_id, "NA", str(e))

if __name__ == "__main__":
    main()