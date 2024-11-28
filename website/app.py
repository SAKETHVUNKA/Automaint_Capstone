import uuid
from flask import Flask, render_template, request, redirect, url_for, session, flash
import firebase_admin
from firebase_admin import credentials, auth, firestore
import requests
import os
import pandas as pd
from datetime import datetime
import boto3
import json
import numpy as np
import time
import tempfile

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with a real secret key in production

cred = credentials.Certificate('./your.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
API_KEY = "your_api_key"

# Configure your AWS S3 credentials
s3_client = boto3.client('s3',
                         aws_access_key_id='your_key_id',
                         aws_secret_access_key='your_secret_access_key',
                         region_name='ap-south-1')

##########################################################################################################################################
########################### FUNCTIONS ####################################################################################################
##########################################################################################################################################

def add_user_to_database(user_id):
    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        'machines': []  # Initialize with an empty machines array
    })
    print(f'User {user_id} added to the database.')

def add_machine_to_database(user_id, machine_id, machine_name):
    user_ref = db.collection('users').document(user_id)
    
    # Check if the user exists
    if not user_ref.get().exists:
        print(f'User {user_id} not found in the database.')
        add_user_to_database(user_id) # there should be no need for the call here but fallback
    
    # Update user's machines array with the new machine
    user_ref.update({
        'machines': firestore.ArrayUnion([{
            'machine_id': machine_id,
            'created_date': datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            'machine_name': machine_name,
            'rul': "NA",
            'datasets': []
        }])
    })
    
    print(f'Machine {machine_id} with details added to user {user_id}.')

def add_dataset_to_database(user_id, machine_id, dataset_id, dataset_name, rul_unit, dataset_filename, categorical_numerical_features=[], status='Should Be Getting Cleaned Now'):
    user_ref = db.collection('users').document(user_id)
    
    # Check if the user exists
    if not user_ref.get().exists:
        print(f'User {user_id} not found in the database.')
        add_user_to_database(user_id) # there should be no need for the call here but fallback

    datasets_ref = user_ref.collection('datasets').document(dataset_id)
    
    dataset_data = {
        'dataset_id': dataset_id,
        'categorical_numerical_features': categorical_numerical_features,
        'dataset_name': dataset_name,
        'rul_unit': rul_unit,
        'dataset_filename': dataset_filename,
        'status': status
    }
    
    dataset_doc = datasets_ref.get()

    if dataset_doc.exists:
        datasets_ref.update({
            'details': firestore.ArrayUnion([dataset_data])
        })
    else:
        datasets_ref.set({
            'details': [dataset_data]
        })

    doc = user_ref.get()

    if doc.exists:
        # Get the current data
        data = doc.to_dict()
        
        # Access the outer array
        outer_array = data.get('machines', [])
        
        # Check if the outer index is valid
        outer_element = next((element for element in outer_array if element.get('machine_id') == machine_id), None)

        if outer_element:
            # Get the inner array at the specified index
            inner_array = outer_element.get('datasets', [])

            # Check if the value is already in the inner array to avoid duplicates
            if dataset_id not in inner_array:
                inner_array.append(dataset_id)

            # Update the outer array with the modified inner array
            outer_element['datasets'] = inner_array
            
            # Update the outer array in the document
            user_ref.update({
                'machines': outer_array
            })
        else:
            print(f"Outer element with ID {machine_id} not found.")
    else:
        print("Document does not exist.")
        
    print(f'Dataset {dataset_id} with details added to user {user_id}.')

def add_model(user_id, dataset_id, dataset_type, model_name, user_given_name, model_id, columns_used=[]):
    user_ref = db.collection('users').document(user_id)
    
    # Check if the user exists
    if not user_ref.get().exists:
        print(f'User {user_id} not found in the database.')
        add_user_to_database(user_id) # there should be no need for the call here but fallback

    model_ref = user_ref.collection('models').document(dataset_id)
    
    model_data = {
        'dataset_id': dataset_id,
        'dataset_type': dataset_type,
        'columns_used' : [], # from json
        'model_id': model_id, # generate
        'user_given_name': user_given_name,
        'model_name': model_name,
        'model_status': "Should be training now.",
        'r2_score': -1,
        'mean_squared_error': -1,
        'mean_absolute_error': -1,
    }
    
    model_doc = model_ref.get()

    if model_doc.exists:
        model_ref.update({
            'details': firestore.ArrayUnion([model_data])
        })
    else:
        model_ref.set({
            'details': [model_data]
        })

def fetch_user_machines(user_id):
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        print(f'User {user_id} not found.')
        return []
    
    user_data = user_doc.to_dict()
    machines = user_data.get('machines', [])
    
    return machines

def fetch_user_datasets(user_id, machine_id):
    user_ref = db.collection('users').document(user_id)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        print(f'User {user_id} not found.')
        return []
    
    user_data = user_doc.to_dict()
    machines = user_data.get('machines', [])
        
    # Check if the outer index is valid
    machine = next((element for element in machines if element.get('machine_id') == machine_id), None)

    # Get the inner array at the specified index
    dataset_ids = machine.get('datasets', [])

    doc_refs = [user_ref.collection('datasets').document(dataset_id) for dataset_id in dataset_ids]

    # Fetch the documents
    docs = db.get_all(doc_refs)

    first_elements = []

    for doc in docs:
        if doc.exists:  # Ensure the document exists
            # Get the details array from the document data
            details = doc.to_dict().get('details', [])  # Default to an empty list if 'details' doesn't exist
            
            if details:  # Check if the details array is not empty
                first_elements.append(details[0]) 
    
    return first_elements

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':

        email = request.form['username']
        password = request.form['password']

        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        # Verify the user's email and password with Firebase Authentication REST API
        response = requests.post(f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}', data=payload)
        data = response.json()

        if 'idToken' in data:
            session['user_id'] = data['localId']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password0 = request.form['password0']
        password1 = request.form['password1']

        if password0 != password1:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('dashboard'))

        payload = {
            "email": username,
            "password": password0,
            "returnSecureToken": True
        }

        # Create a new user with Firebase Authentication REST API
        response = requests.post(f'https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}', data=payload)
        data = response.json()

        if 'idToken' in data:
            session['user_id'] = data['localId']
            add_user_to_database(session['user_id'])
            return redirect(url_for('dashboard'))
        
        else:
            flash(data['error']['message'])

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    machinesArray = fetch_user_machines(session['user_id'])
    
    return render_template('dash.html', machinesArray=machinesArray)

@app.route('/datasets/<machine_id>')
def datasets(machine_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    datasetsArray = fetch_user_datasets(session['user_id'], machine_id)
    
    return render_template('datasets.html', machine_id=machine_id, datasetsArray=datasetsArray)

@app.route('/add_machine', methods=['GET', 'POST'])
def add_machine():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        formData = request.form

        machineName = formData.get('machineName')

        unique_value = str(uuid.uuid4())
            
        # Store the dataset name and options in Firebase
        user_id = session['user_id']
        machine_id = unique_value

        add_machine_to_database(user_id, machine_id, machineName)

        return redirect(url_for('dashboard'))
        
    
    return render_template('add_machine.html')

file_uploaded = False
file_path = None
columns = []
app.config['UPLOAD_FOLDER'] = './uploads'
@app.route('/add_dataset/<machine_id>', methods=['GET', 'POST'])
def add_dataset(machine_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    global file_uploaded, columns, file_path
    
    if request.method == 'POST':
        if file_uploaded:
            formData = request.form
            file_uploaded = False
            columns = []

            datasetName = formData.get('datasetName')
            rulUnit = formData.get('rulUnit')

            numeric_categorical_columns = [feature for feature in formData.keys() if feature.startswith('cat_')]
            # numerical_features = [feature for feature in formData.keys() if feature.startswith('num_')]

            # Remove the prefixes from the feature names if needed
            numeric_categorical_columns = [feature[4:] for feature in numeric_categorical_columns]  # Remove 'cat_' prefix
            # numerical_features = [feature[4:] for feature in numerical_features]      # Remove 'num_' prefix

            dataset_option = formData.get('dataset_option')
    
            status_code = -1
            has_rul = False
            has_unit_number = False
            has_time_cycle_unit = False
            if dataset_option == 'unit_time_cycle':
                has_unit_number = True
                has_time_cycle_unit = True
                has_rul = formData.get('unit_time_cycle_rul') == "yes"
                status_code = 1
            elif dataset_option == 'rul_only':
                has_rul = True
                has_unit_number = formData.get('rul_only_unit_number') is not None
                has_time_cycle_unit = formData.get('rul_only_time_cycle_unit') is not None
                status_code = 2

            # Rename the dataset by appending a unique value
            unique_value = str(uuid.uuid4())
            # filename = os.path.splitext(os.path.basename(file_path))[0] + f'_{unique_value}' + os.path.splitext(file_path)[1]
            filename = f'{unique_value}' + os.path.splitext(file_path)[1]
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.rename(file_path, new_file_path)

            # Upload the dataset to S3
            file_key = f'{filename}'
            with open(new_file_path, 'rb') as data:
                s3_client.upload_fileobj(data, "data-to-clean-automaint", file_key)
            
            # Create a JSON file with the options, including the unique filename
            options = {
                'unique_key': unique_value,
                'actual_machine_id': machine_id,
                'user_id': session['user_id'],
                'dataset_file_name': filename,
                'dataset_type': status_code,

                'datasetName': datasetName,
                'rul_unit': rulUnit,

                # 'categorical_features': categorical_features,
                # 'numerical_features': numerical_features,

                'numeric_categorical_columns': numeric_categorical_columns,
                
                'has_rul': has_rul,
                'has_unit_number': has_unit_number,
                'has_time_cycle_unit': has_time_cycle_unit,

                'cleaned_dataset_bucket': 'data-to-preprocess-automaint',
                "final_dataset_bucket": "data-to-train-automaint",
                "preprocessing_models_bucket": "preprocessing-models-bucket",
                "model_bucket": "ml-models-trained-automaint",

                "user_id": session['user_id'],
            }

            # print(options)
            # print(json.dumps(options, indent=4))

            options_key = f'{os.path.splitext(filename)[0]}_1.json'
            s3_client.put_object(Bucket="data-to-clean-automaint", Key=options_key, Body=json.dumps(options))
            
            # Store the dataset name and options in Firebase
            user_id = session['user_id']
            dataset_id = unique_value
            dataset_filename = filename
            dataset_name = datasetName

            add_dataset_to_database(user_id, machine_id, dataset_id, dataset_name, rulUnit, dataset_filename, categorical_numerical_features=numeric_categorical_columns)

            # Optionally, delete the local file after uploading
            os.remove(new_file_path)

            return redirect(url_for('datasets', machine_id=machine_id))

        if 'file' in request.files:
            file = request.files['file']

            if file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                file_uploaded = True

                file_ext = os.path.splitext(file.filename)[1].lower()
                
                if file_ext == '.txt' or file_ext == '.csv':
                    with open(file_path, 'r') as f:
                        columns = f.readline().strip().split(',')
                
                elif file_ext == '.xlsx':
                    df = pd.read_excel(file_path)
                    columns = df.columns.tolist()
        
    
    return render_template('add_dataset.html', file_uploaded=file_uploaded, columns=columns)

@app.route('/train/<machine_id>/<dataset_id>', methods=['GET', 'POST'])
def train(machine_id, dataset_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_ref = db.collection('users').document(user_id)

    doc_ref = user_ref.collection('datasets').document(dataset_id)

    # Fetch the documents
    doc = doc_ref.get()

    dataset_data = doc.to_dict().get('details', [])[0]

    if "Should Be Getting Cleaned Now" in dataset_data['status']:
        flash(f'Initial Processing not done yet for dataset: {dataset_data["dataset_name"]}')
        return redirect(url_for('datasets', machine_id=machine_id))
    if "Dataset Cleaning is completed ." in dataset_data['status']:
        flash(f'Initial Processing not done yet for dataset: {dataset_data["dataset_name"]}')
        return redirect(url_for('datasets', machine_id=machine_id))
    
    options_key = f"{os.path.splitext(dataset_id)[0]}_3.json"
    json_object = s3_client.get_object(Bucket="data-to-train-automaint", Key=options_key)
    json_data = json.load(json_object['Body'])

    png_image_key = f"{os.path.splitext(dataset_id)[0]}_co-variance_matrix.png" 
    png_image_url = s3_client.generate_presigned_url('get_object',
                                                     Params={'Bucket': 'data-to-preprocess-automaint', 'Key': png_image_key},
                                                     ExpiresIn=3600)  # URL will expire in 1 hour
    
    columns = json_data.get('final_columns', [])
    columns = [column for column in columns if column.lower() != "rul" and column.lower() != "unit_number" and column.lower() != "time_cycle_unit"]

    default_manual_code = """
    import numpy as np
    import pandas as pd
    import sklearn 
    import tensorflow as tf
    import xgboost
    import catboost
    import lightgbm
    import optuna

    def def_and_train_ml(data):# data is the dataframe of the dataset you chose
        # dataset format[for lstm or neural network based models] and split[train and test] 
                
        # model def code 

        # model train code

        # return   model, x_test , y_test, model_type(tensorflow or sklearn[lightbm or catboost or xgboost also to be considered as sklearn]), hyper_parameters(as a dictionary)/\{\}, architecture_of_the_model(as a dictionary)/\{\}, time_steps of dataframe required for one prediction
            """
    
    if request.method == 'POST':
        # Handle training logic here
        formData = request.form

        metadata1 = {}
            
        datasetType = "Pre-Processed Dataset (Dataset after feature_generation and selection is completed)" if formData.get('datasetType') == "preprocessed" else "Cleaned Dataset (Dataset after Cleaning and Scaling)"
        modelType = formData.get('modelType')
        features = [feature[4:] for feature in formData.keys() if feature.startswith("num_")]

        user_given_name = formData.get('userGivenName')
        metadata1['user_given_name'] = user_given_name
        metadata1['dataset_version'] = formData.get('datasetType')
        metadata1['columns_selected'] = bool(features) and formData.get('datasetType') == "cleaned"
        metadata1['columns_selected_for_train'] = features
        metadata1["model_selected_by"] = "manual"
        metadata1['model_selected'] = modelType # Will be manual if custom code

        if modelType == 'manual':
            manual_code = formData.get('manualCode')
            
            if not manual_code:
                flash("You selected Manual mode but did not provide any code.")
                return redirect(url_for('train', dataset_id=dataset_id))

            # Save the manual code to a file
            code_filename = f"{dataset_id}_manual_code_{time.time()}.py"

            # Create a temporary file and store the custom code in it
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
                temp_file.write(manual_code)
                temp_file_path = temp_file.name

            with open(temp_file_path, 'rb') as code_file:
                s3_client.put_object(Bucket='user-def-model-code', Key=code_filename, Body=code_file)

            metadata1['manual_code_s3_key'] = code_filename

            os.remove(temp_file_path)

        if modelType == 'lstm_modifiable':
            # Extract LSTM-specific options from the form
            batch_normalization = 'batchNormalization' in formData
            dropout = 'dropout' in formData
            dropout_rate = formData.get('dropoutRate', type=float) if dropout else None
            num_lstm_layers = formData.get('numLstmLayers', type=int)
            num_dense_layers = formData.get('numDenseLayers', type=int)

            # Add LSTM options to metadata
            metadata1['batch_normalization'] = batch_normalization
            metadata1['dropout'] = dropout
            metadata1['dropout_rate'] = dropout_rate
            metadata1['num_lstm_layers'] = num_lstm_layers
            metadata1['num_dense_layers'] = num_dense_layers

        json_file_obj = s3_client.get_object(Bucket='data-to-train-automaint', Key=options_key)
        json_file_content = json_file_obj['Body'].read().decode('utf-8')
        metadata = json.loads(json_file_content)
        
        metadata1["timestamp"] = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        model_id = modelType + "_" + str(time.time())
        metadata1["model_id"] = model_id

        # print(metadata1)

        metadata.update(metadata1)
        json_key = f"{os.path.splitext(dataset_id)[0]}_3_{time.time()}.json"
        s3_client.put_object(Bucket='data-to-train-automaint', Key=json_key, Body=json.dumps(metadata))

        add_model(user_id, dataset_id, datasetType, modelType, user_given_name, model_id, features)

        return redirect(url_for('list_models', machine_id=machine_id, dataset_id=dataset_id))
    
    return render_template('train.html', columns=columns, default_manual_code=default_manual_code, png_image_url=png_image_url)

@app.route('/<machine_id>/<dataset_id>/models', methods=['GET'])
def list_models(machine_id, dataset_id):
    user_id = session['user_id']

    # Fetch the user's document reference
    user_ref = db.collection('users').document(user_id)
    models_ref = user_ref.collection('models').document(dataset_id)
    
    # Fetch the models document
    models_doc = models_ref.get()
    
    if not models_doc.exists:
        flash('No models for machine yet. Please try again later.')
        print(f'No models found for dataset ID: {dataset_id} under user ID: {user_id}.')
        return redirect(url_for('datasets', machine_id=machine_id, dataset_id=dataset_id))
    
    # Get the details of the models
    models_data = models_doc.to_dict().get('details', [])

    user_doc = user_ref.get()
    user_data = user_doc.to_dict()
    machines = user_data.get('machines', [])
    machine_data = next((m for m in machines if m['machine_id'] == machine_id), None)
    machine_name = machine_data.get("machine_name", "Unnamed Machine")

    details = user_ref.collection('datasets').document(dataset_id).get().to_dict().get('details', [])
    dataset_name = details[0]['dataset_name']

    models_sorted = sorted(models_data, key=lambda x: x['r2_score'], reverse=True)
    best_model = models_sorted[0] if models_sorted else None  # Handle empty case
    
    return render_template(
        'models.html',
        machine_id=machine_id,
        dataset_id=dataset_id, 
        machine_name=machine_name,
        dataset_name=dataset_name,
        models=models_sorted if models_sorted else None,
        best_model=best_model
    )

@app.route('/predict/<machine_id>/<dataset_id>/<model_id>', methods=['GET', 'POST'])
def predict(machine_id, dataset_id, model_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_ref = db.collection('users').document(user_id)
    models_ref = user_ref.collection('models').document(dataset_id)
    models_doc = models_ref.get()
    models_data = models_doc.to_dict().get('details', [])

    for model in models_data:
        if model['model_id'] == model_id:
            if "training is completed" not in model['model_status'] and "Prediction Stopped" not in model['model_status'] and "Predicting" not in model['model_status']:
                flash("Model training incomplete. Please try again later.")
                return redirect(url_for('list_models', machine_id=machine_id, dataset_id=dataset_id))

    user_id = session['user_id']
    models_ref = db.collection('users').document(user_id).collection('models').document(dataset_id)
    models_doc = models_ref.get()   
    details = models_doc.to_dict().get('details', [])
            
    for model in details:
        if "Predicting" in model['model_status']:
            flash("Prediction going on for this machine. Please stop the current prediction and try again later.")
            return redirect(url_for('list_models', machine_id=machine_id, dataset_id=dataset_id))
        if model['model_id'] == model_id:
            # Parameters to include in the JSON payload
            model_file_name = model['model_key']  
            mqtt_topic = f"{dataset_id}_{model_id}/data"
            model_json_key = model['model_metadata_s3_key']
            model_bucket = "ml-models-trained-automaint"

    firebase_update_url = 'https://kbkt311nic.execute-api.ap-south-1.amazonaws.com/default/firebase-update-automaint'
    payload = {
            'user_id': user_id,
            'machine_id': dataset_id,
            'model_id': model_id,
            'operation': 'update',
            'object': 'model',
            'status': 'Predicting',
            'type': "only_status"
    }
    requests.post(firebase_update_url, json=payload)

    # JSON data to send in the POST request
    payload = {
        "model_bucket": model_bucket,
        "model_file_name": model_file_name,
        "mqtt_topic": mqtt_topic,
        "model_json_key": model_json_key,
        'user_id': user_id,
        'machine_id': machine_id,
    }
    # Lambda API URL
    api_url = "https://s79zked5jd.execute-api.ap-south-1.amazonaws.com/default/ec2_predict_automaint"
    
    try:
        print(requests.post(api_url, json=payload))
        return render_template('predict.html', mqtt_topic=mqtt_topic, user_id=user_id, machine_id=dataset_id)
    
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}", 500
    
@app.route('/stop_prediction/<machine_id>/<dataset_id>/<model_id>', methods=['GET'])
def stop_prediction(machine_id, dataset_id, model_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    instance_id = None

    machinesArr = fetch_user_machines(user_id)
    for machine in machinesArr:
        if machine["machine_id"] == machine_id:
            if not machine.get("instance_id", None):
                flash("Prediction hasn't initiated yet. Please try again later.")
                return redirect(url_for('list_models', machine_id=machine_id, user_id=user_id, dataset_id=dataset_id))
            else:
                instance_id = machine.get("instance_id", None)

    firebase_update_url = 'https://kbkt311nic.execute-api.ap-south-1.amazonaws.com/default/firebase-update-automaint'
    payload = {
            'user_id': user_id,
            'machine_id': dataset_id,
            'model_id': model_id,
            'operation': 'update',
            'object': 'model',
            'status': 'Prediction Stopped',
            'type': "only_status"
    }
    requests.post(firebase_update_url, json=payload)

    firebase_update_url = 'https://kbkt311nic.execute-api.ap-south-1.amazonaws.com/default/firebase-update-automaint'
    payload = {
            'user_id': user_id,
            'machine_id': machine_id,
            'operation': 'update',
            'object': 'actual_machine',
            'instance_id': None,
            'type': "instance_id"
        }
    requests.post(firebase_update_url, json=payload)

    api_url = 'https://57dkcai607.execute-api.ap-south-1.amazonaws.com/default/terminate_ec2_instance'
    print(instance_id)

    # The API URL should be the full API Gateway endpoint
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'instance_id': instance_id
    }

    try:
        # Make the POST request to the API Gateway endpoint
        requests.post(api_url, headers=headers, json=payload)
    except:
        return f"An error occurred.", 500
    
    return redirect(url_for('list_models', machine_id=machine_id, dataset_id=dataset_id))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')

if __name__ == '__main__':
    app.run(debug=True, port=9999)