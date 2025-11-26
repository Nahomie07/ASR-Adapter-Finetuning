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

def prepare_dataset(batch, processor: WhisperProcessor, audio_column="audio_filepath", text_column="text", path_prefix=""):
    import torchaudio
    path = batch[audio_column] if path_prefix=="" else os.path.join(path_prefix, batch[audio_column])
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.squeeze(0).numpy()
    inputs = processor.feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with processor.as_target_processor():
        labels = processor.tokenizer(batch[text_column], return_tensors="pt")
    input_values = inputs["input_features"][0]
    label_ids = labels["input_ids"][0]
    return {"input_features": input_values, "labels": label_ids}
