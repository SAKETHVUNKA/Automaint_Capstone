import requests
import argparse

def main():

    api_url = 'terminate_ec2_instance LAMBDA URL'

    parser = argparse.ArgumentParser(description="Terminate EC2 Instance .")
    parser.add_argument('instance_id', type=str, help="Instance ID of the ec2 instance that has to be terminated .")
    args = parser.parse_args()

    # The API URL should be the full API Gateway endpoint
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        'instance_id': args.instance_id
    }

    try:
        # Make the POST request to the API Gateway endpoint
        requests.post(api_url, headers=headers, json=payload)
    
    except Exception as e:
        print(f"Error while making the API request: {str(e)}")

if __name__=="__main__":
    main()