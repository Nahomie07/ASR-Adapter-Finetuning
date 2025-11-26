from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def load_whisper_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    return processor, model
