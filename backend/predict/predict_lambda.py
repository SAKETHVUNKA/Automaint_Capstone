import json
import boto3
import time
import requests
import paramiko 

# Initialize the EC2 and SSM clients
ec2_client = boto3.client('ec2')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    
    metadata = json.loads(event['body'])
    model_bucket = metadata.get('model_bucket')
    model_file_name = metadata.get('model_file_name')
    mqtt_topic = metadata.get('mqtt_topic')
    model_json_key = metadata.get('model_json_key')
    user_id = metadata.get('user_id')
    machine_id = metadata.get('machine_id')
    model_extension = model_file_name.split(".")[-1]
    log_bucket = "automaint-predict"

    # EC2 image details
    ami_id = ''  # Replace with your existing AMI image ID
    instance_type = ''  # Specify the instance type
    key_name = ''  # Specify your key pair for SSH access (if needed)
    
    # Downloading and accessing the private_key
    key_bucket = 'BUCKET CONTAINING THE KEY'
    key_file_key = 'KEY FILE NAME'
    key_file_path = 'TEMPORARY KEY FILE NAME'
    s3_client.download_file(key_bucket, key_file_key, key_file_path)
    private_key = paramiko.RSAKey.from_private_key_file(key_file_path)

    # Paths to the dataset and preprocessing script in S3
    model_s3_path = f's3://{model_bucket}/{model_file_name}'
    predict_ec2_path = f'PREDICT SCRIPT S3 PATH'
    private_key_iot_core = 'IOT CORE PRIVATE KEY S3 PATH'
    certificate_iot_core = 'IOT CORE CERTIFICATE S3 PATH'
    amazon_root_ca_iot_core = 'IOT CORE ROOT CA S3 PATH'

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

    #update instance_id to machine document in firestore
    firebase_update_url = 'PUT FIREBASE UPDATE LAMBDA URL'
    payload = {
            'user_id': user_id,
            'machine_id': machine_id,
            'operation': 'update',
            'object': 'actual_machine',
            'instance_id': instance_id,
            'type': "instance_id"
        }
    print(payload)
    print(instance_id)
    requests.post(firebase_update_url, json=payload)
    
    # Wait for the instance to be running
    print("waiting for ubuntu to start")
    ec2_client.get_waiter('instance_running').wait(InstanceIds=[instance_id])
    time.sleep(30)
    print("ubuntu started")
    
    # Commands to be executed
    commands = [
        "source ec2_env/bin/activate",
        "aws s3 cp {'PREDICT SCRIPT S3 PATH'} /home/ubuntu/startup.sh",
        "chmod +x /home/ubuntu/startup.sh",
        f"nohup /home/ubuntu/startup.sh {model_s3_path} {private_key_iot_core} {certificate_iot_core} {amazon_root_ca_iot_core} {mqtt_topic} {model_json_key} {predict_ec2_path} {model_extension} {log_bucket}"
    ]

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
        'body': json.dumps({"instance_id":instance_id,"status":f'EC2 instance {instance_id} launched, commands executed, and responses fetched.'})
    }