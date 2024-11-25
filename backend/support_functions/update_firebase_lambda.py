import boto3
import json
import firebase_admin
from firebase_admin import credentials, firestore
from typing import List, Dict, Any

class FirebaseManager:
    def __init__(self):
        self.db = None
        self._initialize_firebase()

    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection using AWS Secrets Manager credentials."""
        if not firebase_admin._apps:
            try:
                secrets_manager_client = boto3.client('secretsmanager')
                secret = secrets_manager_client.get_secret_value(SecretId='SECRET ID')
                credentials_json = json.loads(secret['SecretString'])
                cred = credentials.Certificate(credentials_json)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
                raise
        self.db = firestore.client()

    #done
    def update_dataset_status(self, user_id, dataset_id, new_status: str) -> None:
        """Update dataset status using Firestore transaction."""
        print(f"Updating status for Dataset({dataset_id}) to {new_status}...")
        user_ref = self.db.collection('users').document(user_id)
        dataset_ref = user_ref.collection('datasets').document(dataset_id)

        @firestore.transactional
        def update_status_in_transaction(transaction, dataset_ref, new_status):
            dataset_doc = dataset_ref.get(transaction=transaction)
            dataset_data = dataset_doc.to_dict().get("details", [])

            if not dataset_data:
                raise ValueError(f"No details found for Dataset({dataset_id})")

            # Assuming only one dictionary is needed in "details"
            updated_details = [{**dataset_data[0], 'status': new_status}]
            transaction.update(dataset_ref, {'details': updated_details})
            print(f"Dataset({dataset_id}) status updated successfully.")

        transaction = self.db.transaction()
        update_status_in_transaction(transaction, dataset_ref, new_status)

    def update_machine_status(self, user_id, machine_id, new_status: str, new_rul, type, instance_id):
        """Update machine status and RUL using Firestore transaction."""
        print(f"Updating Machine({machine_id}) status to {new_status}...")
        user_ref = self.db.collection('users').document(user_id)

        @firestore.transactional
        def update_machine_in_transaction(transaction, user_ref, machine_id, new_status, new_rul, type, instance_id):
            machine_doc = user_ref.get(transaction=transaction)
                
            machines = machine_doc.to_dict().get('machines', [])
            machine_found = False
            updated_details = []
            
            for machine in machines:
                if machine['machine_id'] == machine_id:
                    machine_found = True
                    if type == "instance_id":
                        updated_details.append({
                            **machine,
                            'instance_id': instance_id
                        })
                    else:
                        updated_details.append({
                            **machine,
                            'status': new_status,
                            'rul': new_rul
                        })
                else:
                    updated_details.append(machine)
                    
            if not machine_found:
                raise ValueError(f"Machine {machine_id} not found")
                
            transaction.update(user_ref, {'machines': updated_details})
            print(f"Status of Machine({machine_id}) updated successfully.")

        transaction = self.db.transaction()
        update_machine_in_transaction(transaction, user_ref, machine_id, new_status, new_rul, type, instance_id)

    def update_model_status(self, user_id, dataset_id, model_id, 
                          new_status: str, r2: float, mae: float, mape: float, mse: float, columns_used: List[str], model_metadata_s3_key: str, model_key: str, type: str, architecture__hyper_parameters) -> None:
        """Update model status and metrics using Firestore transaction."""
        print(f"Updating Model({model_id}) in Dataset({dataset_id}) to {new_status}...")
        machine_ref = self.db.collection('users').document(user_id).collection('models').document(dataset_id)
        
        @firestore.transactional
        def update_model_in_transaction(transaction, machine_ref, new_status, r2, mse, mae, mape, columns_used, model_metadata_s3_key, model_key, type, architecture__hyper_parameters):
            machine_doc = machine_ref.get(transaction=transaction)
                
            details = machine_doc.to_dict().get('details', [])
            model_found = False
            updated_details = []
            
            for model in details:
                if model['model_id'] == model_id:
                    model_found = True
                    if type == "only_status":
                        print("status update")
                        updated_details.append({
                            **model,
                            'model_status': new_status
                        })
                    else:
                        print("all update")
                        updated_details.append({
                            **model,
                            'model_status': new_status,
                            'r2_score': r2,
                            'mean_squared_error': mse,
                            'mean_absolute_error': mae,
                            'architecture/hyper-parameters': architecture__hyper_parameters,
                            'columns_used': columns_used,
                            'model_metadata_s3_key': model_metadata_s3_key,
                            'model_key': model_key
                        })
                else:
                    updated_details.append(model)
                    
            if not model_found:
                raise ValueError(f"Model {model_id} not found")
                
            transaction.update(machine_ref, {'details': updated_details})
            print(f"Status of Model({model_id}) in Dataset({dataset_id}) updated successfully.")

        transaction = self.db.transaction()
        update_model_in_transaction(transaction, machine_ref, new_status, r2, mse, mae, mape, columns_used, model_metadata_s3_key, model_key, type, architecture__hyper_parameters)

    def add_model(self, user_id, dataset_id, dataset_type: str, 
                 model_name: str, model_id: str, user_given_name: str) -> None:
        """Add a new model using Firestore transaction."""
        print(f"Creating Model({model_id}) in Machine({dataset_id})...")
        machine_ref = self.db.collection('users').document(user_id).collection('models').document(dataset_id)

        model_data = {
            'dataset_name': dataset_id,
            'dataset_type': dataset_type,
            'columns_used': [],
            'model_id': model_id,
            'user_given_name': user_given_name,
            'model_name': model_name,
            'model_status': "Should be training now.",
            'r2_score': -1,
            'mean_squared_error': -1,
            'mean_absolute_error': -1,
            'mean_absolute_percentage_error': -1,
            'model_metadata_s3_key': "NA",
            'model_key': "NA"
        }
        
        @firestore.transactional
        def add_model_in_transaction(transaction, machine_ref, model_data):
            machine_doc = machine_ref.get(transaction=transaction)
            if not machine_doc.exists:
                transaction.set(machine_ref, {'details': [model_data]})
            else:
                # Check if model_id already exists
                existing_details = machine_doc.to_dict().get('details', [])
                if any(model['model_id'] == model_id for model in existing_details):
                    raise ValueError(f"Model {model_id} already exists")
                transaction.update(machine_ref, {'details': firestore.ArrayUnion([model_data])})
            print(f"Model({model_id}) created successfully in Dataset({dataset_id}).")

        transaction = self.db.transaction()
        add_model_in_transaction(transaction, machine_ref, model_data)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for Firebase operations."""
    try:
        body = json.loads(event['body'])
        user_id = body.get('user_id')
        dataset_id = body.get('machine_id')
        operation = body.get('operation')
        object_type = body.get('object')
        
        if not all([user_id, dataset_id, operation, object_type]):
            return {
                'statusCode': 400,
                'body': json.dumps('Missing required parameters')
            }

        firebase_manager = FirebaseManager()

        if operation == "update":
            if object_type == "machine":
                status = body.get('status')
                if not status:
                    raise ValueError("Status is required for machine update")
                firebase_manager.update_dataset_status(user_id, dataset_id, status)
            
            elif object_type == "model":
                model_id = body.get('model_id')
                status = body.get('status')
                if not all([model_id, status]):
                    raise ValueError("model_id and status are required for model update")
                firebase_manager.update_model_status(
                    user_id, dataset_id, model_id, status,
                    body.get('r2', -1), body.get('mae', -1), body.get('mape', -1), body.get('mse', -1), body.get('columns_used'), body.get('model_metadata_s3_key',"NA"), body.get('model_key',"NA"), body.get('type',"all"), body.get("architecture/hyper-parameters",{})
                )
            
            elif object_type == "actual_machine":
                status = body.get('status'," ")
                rul = body.get('rul',-1)
                # if not all([rul, status]):
                #     raise ValueError("rul and status are required for model update")
                print(body)
                firebase_manager.update_machine_status(user_id, dataset_id, status, rul, body.get('type',"all"), body.get('instance_id',"NA"))
                
        elif operation == "create" and object_type == "model":
            required_fields = ['dataset_type', 'model_name', 'model_id','user_given_name']
            if not all(body.get(field) for field in required_fields):
                raise ValueError(f"Missing required fields: {required_fields}")
            
            firebase_manager.add_model(
                user_id, dataset_id,
                body['dataset_type'],
                body['model_name'],
                body['model_id'],
                body['user_given_name'],
            )

        return {
            'statusCode': 200,
            'body': json.dumps(f'Operation {operation} on {object_type} successful.')
        }

    except ValueError as ve:
        return {
            'statusCode': 400,
            'body': json.dumps(f'Validation error: {str(ve)}')
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('Internal server error')
        }