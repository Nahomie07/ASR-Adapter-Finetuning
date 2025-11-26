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
    import torchaudio

    # Charger l'audio
    waveform, sr = torchaudio.load(sample["audio_filepath"])

    # Resampler si nécessaire (Whisper attend 16kHz)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Features audio (input_values)
    input_features = processor.feature_extractor(
        waveform.squeeze(0), sampling_rate=16000, return_tensors="pt"
    ).input_features.squeeze(0)

    # Labels texte
    labels = processor.tokenizer(sample.get("text", ""), return_tensors="pt").input_ids.squeeze(0)

    return {"input_features": input_features, "labels": labels}

