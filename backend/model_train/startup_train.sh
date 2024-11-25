#!/bin/bash

# Accept S3 paths as arguments
DATASET_S3_PATH=$1
TRAINING_SCRIPT_S3_PATH=$2
EC2_TERMINATE_S3_PATH=$3
INSTANCE_ID=$4
MODEL_BUCKET=$5
UNIQUE_KEY=$6
JSON_BUCKET_NAME=$7
JSON_KEY=$8
MODEL_ID=$9

#create log file
touch info.log

# Fetch the dataset and training script and ec2_terminate script from S3
aws s3 cp "$DATASET_S3_PATH" /home/ubuntu/dataset.csv >> info.log 2>&1
echo "--> Dataset downloaded" >> info.log
aws s3 cp "$TRAINING_SCRIPT_S3_PATH" /home/ubuntu/train_script.py >> info.log 2>&1
echo "--> Training script downloaded" >> info.log
aws s3 cp "$EC2_TERMINATE_S3_PATH" /home/ubuntu/terminate.py >> info.log 2>&1
echo "--> Termination script downloaded" >> info.log

# Execute the training script
python3 -W ignore /home/ubuntu/train_script.py "$JSON_BUCKET_NAME" "$JSON_KEY" "$MODEL_ID">> info.log 2>&1
echo "--> Train script executed" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${MODEL_BUCKET}/${UNIQUE_KEY}__${MODEL_ID}__train_logfile.log"

# Execute the ec2 termination script
python3 -W ignore /home/ubuntu/terminate.py "$INSTANCE_ID" >> info.log 2>&1
echo "--> Termination script executed" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${MODEL_BUCKET}/${UNIQUE_KEY}__${MODEL_ID}__train_logfile.log"