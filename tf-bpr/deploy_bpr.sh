#!/usr/bin/env bash


# Define a timestamp function
timestamp() {
  date +"%I_%M_%S"
}


export NAME=snado_amzn_bpr_$(timestamp)_$1
echo $NAME


gcloud beta ml jobs submit training $NAME \
--package-path=amzn/bpr \
--module-name=bpr.bpr \
--staging-bucket="gs://sharknado-team" \
--region=us-east1 \
--scale-tier=BASIC_GPU