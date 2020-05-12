#!/bin/bash

echo "Making data directories."

declare -a SUB_DIRECTORIES=("model" "predictions" "source")
if [ ! -d "data" ]; then
  mkdir "data/"

  for DIRECTORY in "${SUB_DIRECTORIES[@]}"; do
    mkdir "data/$DIRECTORY"
  done
fi


echo "Installing dependencies"
sudo apt install -y python3.7 python3-pip build-essential libssl-dev libffi-dev python-dev python3-venv

echo "Making virtual environment"
python3 -m venv ".env"
source ".env/bin/activate"

echo "Installing python libraries"
pip install -r "requirements.txt"
