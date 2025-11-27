import torchaudio
import torch

SAMPLE_RATE = 16000

def prepare_dataset(sample, processor):
    # Charger audio
    waveform, sr = torchaudio.load(sample["audio_filepath"])

    # Convertir en mono si nécessaire
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resampler si nécessaire
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

    # Convertir en numpy 1D
    waveform = waveform.squeeze(0).numpy()

    # Feature extraction
    input_features = processor.feature_extractor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.squeeze(0)

    # Tokenizer labels
    labels = processor.tokenizer(
        sample.get("text", ""), return_tensors="pt"
    ).input_ids.squeeze(0)

    return {"input_features": input_features, "labels": labels}


def collate_fn(batch):
    input_feats = [b["input_features"] for b in batch]
    # padding dynamique, converti en 2D si nécessaire
    input_feats = [f if f.ndim == 2 else f.unsqueeze(0) for f in input_feats]
    input_feats = torch.nn.utils.rnn.pad_sequence(input_feats, batch_first=True, padding_value=0.0)

    labels = [b["labels"] for b in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_features": input_feats, "labels": labels}
