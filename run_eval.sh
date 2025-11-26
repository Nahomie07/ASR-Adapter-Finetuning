#!/bin/bash
# Generate base transcriptions
python src/evaluate.py --model_name openai/whisper-small --adapter_path None --split test --out base_transcriptions.txt
# Generate finetuned transcriptions
python src/evaluate.py --model_name openai/whisper-small --adapter_path ./adapters/adapter_weights.pth --split test --out finetuned_transcriptions.txt
