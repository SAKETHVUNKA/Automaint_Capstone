import boto3
import json
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO,BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path, has_headers=True):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        data = pd.read_csv(file_path)
    elif file_extension == 'txt':
        with open(file_path, 'r') as file:
            first_line = file.readline()
            if ',' in first_line:
                separator = ','
            else:
                separator = ' ' 
        data = pd.read_csv(file_path, sep=separator)
    elif file_extension == 'xlsx':
        data = pd.read_excel(file_path)
    return data

def remove_nan_columns(df):
    nan_columns = [col for col in df.columns if df[col].isna().sum() > (len(df) / 2)]
    df.drop(columns=nan_columns, inplace=True)
    return df

def adding_rul(data, has_rul):
    if has_rul:
        data.rename(columns={data.columns[-1]: 'RUL'}, inplace=True)
        return data
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_cycle_unit'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_cycle_unit']
    df.drop(columns=['max'],inplace = True)
    return df

def drop_rows_with_nan(data):
    data = data.drop_duplicates()
    data.replace('', pd.NA, inplace=True)
    df_cleaned = data.dropna()
    return df_cleaned

def modify_column_names(df, has_unit_number, has_time_cycle_unit):
    new_columns = df.columns.tolist()
    if has_unit_number:
        new_columns[0] = 'unit_number'
    if has_time_cycle_unit:
        if has_unit_number:
            new_columns[1] = 'time_cycle_unit'
        else:
            new_columns[0] = 'time_cycle_unit'
    df.columns = new_columns
    return df

def scale_and_encode_data(df,exclude_columns):
    exclude_columns.extend(['unit_number', 'time_cycle_unit','RUL'])
    scaler = StandardScaler()
    label_encoders = {}
    transformations = {}

    # Process each column
    for column in df.columns:
        if column in exclude_columns:
            continue
        elif df[column].dtype == 'object':
            # Check if the column is categorical (text)
            if df[column].nunique() < len(df[column]):
                # Encode categorical columns
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
                transformations[column] = {
                    'type': 'encoding',
                    'conversion': dict(zip(le.classes_, le.transform(le.classes_)))
                }
        elif pd.api.types.is_numeric_dtype(df[column]):
            # Scale numerical columns
            scaler.fit(df[[column]])
            df[column] = scaler.transform(df[[column]])
            transformations[column] = {
                'type': 'scaling',
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
    info = {
        'transformations': transformations
    }
    return df, info

def lambda_handler(event, context):
    try:
        s3_client = boto3.client('s3')

        # Extracting bucket name and key from the event when a JSON file is uploaded
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        # Getting the JSON file content from S3
        json_file_obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        json_file_content = json_file_obj['Body'].read().decode('utf-8')
        metadata = json.loads(json_file_content)

        # Extracting parameters from the JSON content
        unique_key = metadata.get('unique_key')
        dataset_file_name = metadata.get('dataset_file_name')
        dataset_type = metadata.get('dataset_type')
        numeric_categorical_columns = metadata.get('numeric_categorical_columns')
        output_bucket_name = metadata.get('cleaned_dataset_bucket')
        user_id = metadata.get('user_id') 
        if dataset_type == 1:    
            has_rul = metadata.get("has_rul")
            has_unit_number = True
            has_time_cycle_unit = True
        else:
            has_rul = True
            has_unit_number = metadata.get("has_unit_number")
            has_time_cycle_unit = metadata.get("has_time_cycle_unit")

        # Download the dataset file from the specified S3 bucket
        dataset_download_path = '/tmp/' + dataset_file_name
        s3_client.download_file(bucket_name, dataset_file_name, dataset_download_path)

        # Load and process the dataset
        df = load_data(dataset_download_path)
        df_proper_columns = modify_column_names(df,has_unit_number,has_time_cycle_unit)
        df_proper_columns_without_null_columns = remove_nan_columns(df_proper_columns)
        rul_data = adding_rul(df_proper_columns_without_null_columns,has_rul)
        clean_data = drop_rows_with_nan(rul_data)
        scaled_data, transformations_info = scale_and_encode_data(clean_data, numeric_categorical_columns)

        # Convert the clean data to CSV format and upload it to S3
        csv_buffer = StringIO()
        scaled_data.to_csv(csv_buffer, index=False)
        output_key = unique_key + '_cleaned_dataset.csv'
        s3_client.put_object(Bucket=output_bucket_name, Key=output_key, Body=csv_buffer.getvalue())

        # Generating and uploading the co-variance matrix of cleaned dataset
        ## Calculating the covariance matrix of the scaled dataset
        cov_matrix = scaled_data.cov()
        ## Create a heatmap of the covariance matrix
        plt.figure(figsize=(25, 15),dpi=65)
        sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm',vmin = -1000,vmax= 1000)
        ## Save the heatmap as a PNG file
        plt.savefig('/tmp/heatmap.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        ## Uploading co-variance matrix to S3
        png_key = unique_key + "_co-variance_matrix.png"
        s3_client.upload_file('/tmp/heatmap.png', output_bucket_name, png_key)

        # Convert the transformation information to JSON format and uploading it to S3
        transformation_info_key = unique_key + "_2.json"
        final_columns = scaled_data.columns.to_list()
        cleaned_dataset_name = output_key
        metadata1 = {}
        metadata1["transformations_info"] = transformations_info
        metadata1["final_columns"]  = final_columns
        metadata1["cleaned_dataset_name"] = cleaned_dataset_name
        metadata.update(metadata1)
        s3_client.put_object(Bucket=output_bucket_name,Key=transformation_info_key,Body=json.dumps(metadata))

        # Update status variable
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': unique_key,
            'operation': 'update',
            'object': 'machine',
            'status': 'Dataset Cleaning is completed .'
        }
        requests.post(firebase_update_url, json=payload)

        return {
            'statusCode': 200,
            'body': json.dumps(f'Data processing and upload complete to bucket {output_bucket_name}.')
        }

    except Exception as e:

        # Update status variable
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': unique_key,
            'operation': 'update',
            'object': 'machine',
            'status': f'Error in Dataset Cleaning : {str(e)}'
        }
        requests.post(firebase_update_url, json=payload)

        print(f"Error Cleaning Dataset : {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"An error occurred: {e}")
        }