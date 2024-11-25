import json
import boto3
import time
import paramiko
import requests

# Initialize the EC2 and SSM clients
ec2_client = boto3.client('ec2')
s3_client = boto3.client('s3')

def lambda_handler(event, context):

    # Extracting bucket name and key from the event when a JSON file is uploaded
    in_json_bucket_name = event['Records'][0]['s3']['bucket']['name']
    in_json_key = event['Records'][0]['s3']['object']['key']

    # Getting the JSON file content from S3
    json_file_obj = s3_client.get_object(Bucket=in_json_bucket_name, Key=in_json_key)
    json_file_content = json_file_obj['Body'].read().decode('utf-8')
    metadata = json.loads(json_file_content)
    unique_key = metadata.get("unique_key")
    model_bucket = metadata.get("model_bucket")
    model_selected_by = metadata.get("model_selected_by")
    model_name = metadata.get("model_selected","svr_simple")
    model_id = metadata.get('model_id',model_name+"_"+str(time.time()))

    # EC2 image details
    ami_id = ''  # Replace with your existing AMI image ID
    instance_type = ''  # Specify the instance type
    key_name = 'KEY NAME'  # Specify your key pair for SSH access (if needed)
    
    # Downloading and accessing the private_key
    key_bucket = 'KEY CONTAINING BUCKET'
    key_file_key = 'KEY FILE NAME'
    key_file_path = 'TEMP KEY FILE NAME'
    s3_client.download_file(key_bucket, key_file_key, key_file_path)
    private_key = paramiko.RSAKey.from_private_key_file(key_file_path)

    # Path to the dataset in S3
    if model_selected_by == "automatic":
        final_dataset_bucket = metadata.get("final_dataset_bucket")
        final_dataset_name = metadata.get("final_dataset_name")
        dataset_s3_path = f's3://{final_dataset_bucket}/{final_dataset_name}'
    elif model_selected_by == "manual":
        dataset_version = metadata.get("dataset_version")
        if dataset_version == "preprocessed":
            final_dataset_bucket = metadata.get("final_dataset_bucket")
            final_dataset_name = metadata.get("final_dataset_name")
            dataset_s3_path = f's3://{final_dataset_bucket}/{final_dataset_name}'
        elif dataset_version == "cleaned":
            final_dataset_bucket = metadata.get("cleaned_dataset_bucket")
            final_dataset_name = metadata.get("cleaned_dataset_name")
            dataset_s3_path = f's3://{final_dataset_bucket}/{final_dataset_name}'

    if model_selected_by == "automatic":
        user_id = metadata.get('user_id')
        dataset_type = "Pre-Processed Dataset (Dataset after feature_generation and selection is completed) ."
        # columns_used = metadata.get('selected_columns')
        firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
        payload = {
            'user_id': user_id,
            'machine_id': unique_key,
            'operation': "create",
            'object': "model",
            'model_id': model_id,
            'model_name': model_name,
            'user_given_name': f"{model_name}_Automatic",
            'dataset_type': dataset_type,
            'columns_used': [],
            'status': 'Should be training now .'
        }
        requests.post(firebase_update_url, json=payload)

    # Paths to the training script and termination script in S3
    training_script_s3_path = f's3://{"MODEL TEMPLATES BUCKET"}/{model_name}.py'
    terminate_ec2_script_s3_path = f'EC2 TERMINATION TRIGGER SCRIPT S3 PATH'
    
    # Launch the EC2 instance
    instances = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={'Arn': "IAM INSTANCE PROFILE"},
        SecurityGroupIds=['SECURITY GROUP ID']
    )
    instance_id = instances['Instances'][0]['InstanceId']
    print(f"instance started:{instance_id}")
    
    # Wait for the instance to be running
    print("waiting for ubuntu to start")
    ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
    time.sleep(30)
    print("ubuntu started")

    print(f"/home/ubuntu/startup.sh {dataset_s3_path} {training_script_s3_path} {terminate_ec2_script_s3_path} {instance_id} {model_bucket} {unique_key} {in_json_bucket_name} {in_json_key} {model_id}")

    # Commands to be executed
    commands = [
        "source ec2_env/bin/activate",
        "aws s3 cp {'STARTUP SCRIPT S3 PATH'} /home/ubuntu/startup.sh",
        "chmod +x /home/ubuntu/startup.sh",
        f"nohup /home/ubuntu/startup.sh {dataset_s3_path} {training_script_s3_path} {terminate_ec2_script_s3_path} {instance_id} {model_bucket} {unique_key} {in_json_bucket_name} {in_json_key} {model_id}"
    ]

    if model_name == "manual":
        manual_code_file_name = metadata.get("manual_code_s3_key")
        user_code_download_command = f"aws s3 cp s3://{'MANUAL CODE BUCKET NAME'}/{manual_code_file_name} /home/ubuntu/ml_def_and_train_module.py"
        commands.insert(2, user_code_download_command)

    # EC2 instance username and public ip address
    ec2_user = 'ubuntu'
    instances1 = ec2_client.describe_instances(InstanceIds=[instance_id])
    ec2_ip = instances1['Reservations'][0]['Instances'][0]['PublicIpAddress']
    print(f"ec2 public ip_address : {ec2_ip}")
    
    # Establishing SSH connection
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ec2_ip, username=ec2_user, pkey=private_key)
    shell = client.invoke_shell()

    # Execute the commands one at a time
    for command in commands:
        try:
            shell.send(command+" \n")
            time.sleep(10)
            print(f"Command : {command} executed .")
                
        except Exception as e:
                    print(f"Error executing command '{command}': {e}")
                    ec2_client.terminate_instances(InstanceIds=[instance_id])
                    return {
                        'statusCode': 500,
                        'body': json.dumps(f'Error executing command: {str(e)}')
                    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'EC2 instance {instance_id} launched, commands executed, and responses fetched.')
    }