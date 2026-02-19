#!/bin/bash
# Helper script to run the OpenClawPyLite bot within its virtual environment

if [ -d "venv" ]; then
    echo "Starting bot using virtual environment..."
    ./venv/bin/python3 bot.py
else
    echo "Error: venv directory not found."
    echo "Please create a virtual environment first: python3 -m venv venv"
    echo "Then install dependencies: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi
