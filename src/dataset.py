import os
from datasets import load_dataset
import torchaudio
import torch
from dataclasses import dataclass
from typing import List, Dict
from transformers import WhisperProcessor

SAMPLE_RATE = 16000

def download_dataset(destination_dir="./data"):
    ds = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")
    ds.save_to_disk(destination_dir)
    return ds

def speech_file_to_array_fn(batch, path_prefix="", audio_column="audio_filepath"):
    path = os.path.join(path_prefix, batch[audio_column]) if path_prefix else batch[audio_column]
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return {"speech": waveform.squeeze(0).numpy(), "sampling_rate": SAMPLE_RATE}

def prepare_dataset(sample, processor):
    # Charger l'audio
    import torchaudio
    waveform, sr = torchaudio.load(sample["audio_filepath"])
    
    # Features audio
    input_features = processor.feature_extractor(
        waveform.squeeze(0), sampling_rate=sr, return_tensors="pt"
    ).input_features.squeeze(0)
    
    # Labels texte
    labels = processor.tokenizer(sample["text"], return_tensors="pt").input_ids.squeeze(0)
    
    return {"input_features": input_features, "labels": labels}
