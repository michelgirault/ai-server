#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#install additional lib
pip install xformers

#start app server api
uvicorn app:app --host 0.0.0.0 --port 4000