#!/bin/bash

# Run the fixed extreme boost training script
echo "Starting fixed extreme boost training..."
python run_extreme_boost_v2.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed. Please check the logs for errors."
fi