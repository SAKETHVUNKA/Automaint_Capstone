import json
import boto3
import time
import paramiko 

# Initialize the EC2 and SSM clients
ec2_client = boto3.client('ec2')
s3_client = boto3.client('s3')

def lambda_handler(event, context):

    # Extracting bucket name and key from the event when a JSON file is uploaded
    in_json_bucket_name = event['Records'][0]['s3']['bucket']['name']
    in_json_key = event['Records'][0]['s3']['object']['key']
    print(in_json_bucket_name,in_json_key)

    # Getting the JSON file content from S3
    json_file_obj = s3_client.get_object(Bucket=in_json_bucket_name, Key=in_json_key)
    json_file_content = json_file_obj['Body'].read().decode('utf-8')
    metadata = json.loads(json_file_content)
    unique_key = metadata.get('unique_key')
    output_bucket_name = metadata.get("cleaned_dataset_bucket")
    final_dataset_bucket = metadata.get("final_dataset_bucket")
    cleaned_dataset_name = metadata.get("cleaned_dataset_name")

    # EC2 image details
    ami_id = ""  # Replace with your existing AMI image ID
    instance_type = ''  # Specify the instance type
    key_name = ''  # Specify your key pair for SSH access (if needed)
    
    # Downloading and accessing the private_key
    key_bucket = 'BUCKET CONTAINING THE KEY'
    key_file_key = 'KEY FILE NAME'
    key_file_path = 'TEMPORARY KEY FILE NAME'
    s3_client.download_file(key_bucket, key_file_key, key_file_path)
    private_key = paramiko.RSAKey.from_private_key_file(key_file_path)

    # Paths to the dataset and preprocessing script in S3
    dataset_s3_path = f's3://{output_bucket_name}/{cleaned_dataset_name}'
    training_script_s3_path = 'PREPROCESSING PYTHON SCRIPT S3 PATH'
    terminate_ec2_script_s3_path = 'TERMINATION TRIGGER PYTHON SCRIPT S3 PATH'
    
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
    
    # Commands to be executed
    commands = [
        "source ec2_env/bin/activate",
        "aws s3 cp {'STARTUP SCRIPT S3 PATH'} /home/ubuntu/startup.sh",
        "chmod +x /home/ubuntu/startup.sh",
        f"nohup /home/ubuntu/startup.sh {unique_key} {dataset_s3_path} {training_script_s3_path} {terminate_ec2_script_s3_path} {final_dataset_bucket} {in_json_bucket_name} {in_json_key} {instance_id}"
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
        'body': json.dumps(f'EC2 instance {instance_id} launched, commands executed, and responses fetched.')
    }