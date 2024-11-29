#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#download models for sd
#install req for the app and server sd
pip install -r requirements.txt

#start app server api
uvicorn app:app