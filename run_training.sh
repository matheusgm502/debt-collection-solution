#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root directory
cd "$SCRIPT_DIR"

# Create output directory if it doesn't exist
mkdir -p output

# Run the training and evaluation script
echo "Running training and evaluation..."
python src/train_and_evaluate.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Training and evaluation completed successfully!"
    echo "You can now run the Streamlit dashboard with: ./src/app/run_app.sh"
else
    echo "Training and evaluation failed. Please check the error messages above."
fi 