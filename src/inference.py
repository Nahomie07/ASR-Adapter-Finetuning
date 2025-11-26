# Small wrapper to run evaluation for base and finetuned models.
import argparse
import os
from evaluate import generate_transcriptions
from evaluate import load_base_model
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base","finetuned"], default="base")
    parser.add_argument("--adapter_path", default="./adapters/adapter_weights.pth")
    parser.add_argument("--out", default="base_transcriptions.txt")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    ds = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")[args.split]
    processor, model = load_base_model()
    if args.mode == "base":
        generate_transcriptions(processor, model, ds, adapter_weights_path=None, device="cuda" if False else "cpu", output_path=args.out)
    else:
        generate_transcriptions(processor, model, ds, adapter_weights_path=args.adapter_path, device="cuda" if False else "cpu", output_path=args.out)
