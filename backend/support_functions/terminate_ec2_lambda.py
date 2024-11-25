import boto3
import json

def lambda_handler(event, context):
    # Parse the instance ID from the incoming API request
    try:
        body = json.loads(event['body'])
        instance_id = body['instance_id']
    except (KeyError, TypeError) as e:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Invalid input, instance_id is required'
            })
        }
    
    # Create an EC2 client
    ec2 = boto3.client('ec2')

    try:
        # Terminate the instance
        response = ec2.terminate_instances(InstanceIds=[instance_id])
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f"Terminated instance: {instance_id}",
                'response': response
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Error terminating instance: {str(e)}"
            })
        }