#!/bin/bash
# This shell script will run the experiments described in the readme file
declare -a CLASSIFIERS=("cnn" "mlp" "rnn")
declare -a FEATURES=("all" "controller" "controller+doe" "sensors" "sensors+doe")

for classifier in "${CLASSIFIERS[@]}"
do
  for feature_set in "${FEATURES[@]}"
  do
    echo "Now Running $feature_set $classifier $it"
    python main.py "$feature_set" "$classifier"
  done
done