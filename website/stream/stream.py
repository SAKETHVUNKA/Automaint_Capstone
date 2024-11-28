from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json
import pandas as pd
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MQTT connection details
CLIENT_ID = f"DataFromPythonScript_{timestamp}"
ENDPOINT = "your_endpoint"
TOPIC = "provided_topic"
CERT_PATH = "./your-certificate.pem.crt"
KEY_PATH = "./your-private.pem.key"
ROOT_CA_PATH = "./AmazonRootCA1.pem"

# Initialize MQTT client
mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
mqtt_client.configureEndpoint(ENDPOINT, 8883)
mqtt_client.configureCredentials(ROOT_CA_PATH, KEY_PATH, CERT_PATH)
mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite offline publish queueing
mqtt_client.configureDrainingFrequency(2)  # Draining frequency for publish queueing
mqtt_client.configureConnectDisconnectTimeout(30)  # 10 sec
mqtt_client.configureMQTTOperationTimeout(15)  # 5 sec

# Connect to the MQTT broker
mqtt_client.connect()
print("Connected to MQTT broker.")

# Load the CSV file
file_path = "your_dataset.csv"  # Adjust the path as necessary
df = pd.read_csv(file_path)

# Exclude specific columns
df = df.drop(columns=["machine_number", "sesseion_id"], errors='ignore')
df = df.dropna()

# Publish each row as a JSON message
for _, row in df.iterrows():
    
    message = input("enter :")
    if message == "stop":
        print("Stopped by user")
        break

    payload = row.to_dict()  # Convert row to dictionary format
    mqtt_client.publish(TOPIC, json.dumps(payload), 1)
    print("Message published:", payload)

print("All messages sent!")
mqtt_client.disconnect()