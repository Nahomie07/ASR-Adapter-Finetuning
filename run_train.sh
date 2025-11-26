#!/bin/bash
# Example run script. Edit batch size / epochs as needed.
python src/train.py --model_name openai/whisper-small --batch_size 8 --num_epochs 3 --bottleneck_dim 64 --adapter_dir ./adapters
