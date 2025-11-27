from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def load_whisper_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language='kin', task="transcribe"
    )
    return processor, model
