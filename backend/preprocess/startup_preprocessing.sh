#!/bin/bash

# Accept S3 paths as arguments
UNIQUE_KEY=$1
DATASET_S3_PATH=$2
TRAINING_SCRIPT_S3_PATH=$3
EC2_TERMINATE_S3_PATH=$4
FINAL_DATASET_BUCKET=$5
IN_JSON_BUCKET=$6
IN_JSON_KEY=$7
INSTANCE_ID=$8

#create log file
touch info.log

# Fetch the dataset and training script from S3
aws s3 cp "$DATASET_S3_PATH" /home/ubuntu/dataset.csv  >> info.log 2>&1
echo "--> Dataset downloaded" >> info.log
aws s3 cp "$TRAINING_SCRIPT_S3_PATH" /home/ubuntu/train_script.py  >> info.log 2>&1
echo "--> Training script downloaded" >> info.log
aws s3 cp "$EC2_TERMINATE_S3_PATH" /home/ubuntu/terminate.py  >> info.log 2>&1
echo "--> Termination script downloaded" >> info.log

# Execute the training script
python3 -W ignore /home/ubuntu/train_script.py "$UNIQUE_KEY" "$IN_JSON_BUCKET" "$IN_JSON_KEY" >> info.log 2>&1
echo "--> Preprocessing script executed" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${FINAL_DATASET_BUCKET}/${UNIQUE_KEY}_preprocess_logfile.log"

# Execute the ec2 termination script
python3 -W ignore /home/ubuntu/terminate.py "$INSTANCE_ID" >> info.log 2>&1
echo "--> Termination script executed" >> info.log

#push log file to s3
aws s3 cp info.log "s3://${FINAL_DATASET_BUCKET}/${UNIQUE_KEY}_preprocess_logfile.log"