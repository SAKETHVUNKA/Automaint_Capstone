#!/bin/bash

# Accept S3 paths as arguments
MODEL_S3_PATH=$1
PRIVATE_KEY_IOT_CORE=$2
CERTIFICATE_IOT_CORE=$3
AMAZON_ROOT_CA_IOT_CORE=$4
MQTT_TOPIC=$5
MODEL_JSON_KEY=$6
PREDICT_EC2_PATH=$7
MODEL_EXTENSION=$8
LOG_BUCKET=$9

#create log file
touch info.log

# Fetch the dataset and training script from S3
aws s3 cp "$PREDICT_EC2_PATH" /home/ubuntu/train_script.py  >> info.log 2>&1
echo "--> Prediction script downloaded" >> info.log
aws s3 cp "$MODEL_S3_PATH" /home/ubuntu/model."$MODEL_EXTENSION"  >> info.log 2>&1
echo "--> Model file downloaded" >> info.log
aws s3 cp "$PRIVATE_KEY_IOT_CORE" /home/ubuntu/private_key_iot_core.pem.key  >> info.log 2>&1
echo "--> Private Key downloaded" >> info.log
aws s3 cp "$CERTIFICATE_IOT_CORE" /home/ubuntu/certificate_iot_core.pem.crt  >> info.log 2>&1
echo "-->  Certificate downloaded" >> info.log
aws s3 cp "$AMAZON_ROOT_CA_IOT_CORE" /home/ubuntu/root_ca_iot_core.pem  >> info.log 2>&1
echo "--> AWS Root CA downloaded" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${LOG_BUCKET}/${MQTT_TOPIC}_preprocess_logfile.log"

# Execute the training script
python3 -W ignore /home/ubuntu/train_script.py "$MODEL_JSON_KEY" "$MQTT_TOPIC" >> info.log 2>&1
echo "--> Prediction script executed" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${LOG_BUCKET}/${MQTT_TOPIC}_preprocess_logfile.log"