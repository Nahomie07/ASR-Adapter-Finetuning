import torch
import torchaudio
import tarfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from jiwer import wer
import os
import argparse

class TarAudioDataset:
    """
    Dataset qui lit des fichiers audio directement à partir d'un tarball.
    """
    def __init__(self, tar_path, n_samples=None):
        self.samples = []
        self.tar_path = tar_path
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith((".webm",".wav"))]
            if n_samples:
                members = members[:n_samples]
            for m in members:
                self.samples.append(m.name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"audio_file": self.samples[idx], "text": ""}  # text vide si pas dispo

    def get_waveform(self, idx):
        """
        Retourne waveform et sample rate directement depuis le tarball.
        """
        sample_name = self.samples[idx]
        with tarfile.open(self.tar_path, "r") as tar:
            f = tar.extractfile(sample_name)
            waveform, sr = torchaudio.load(f)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        return waveform.squeeze(0).numpy()

# ------------------------------
# Fonctions de génération
# ------------------------------
def load_base_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def load_adapter_weights(model, adapter_path):
    adapter_state = torch.load(adapter_path, map_location="cpu")
    model_state = model.state_dict()
    for k,v in adapter_state.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Loaded adapter weights from {adapter_path}")

def generate_transcriptions_from_tar(processor, model, dataset, tar_path, device="cuda", adapter_path=None, output_path="transcriptions.txt"):
    if adapter_path:
        load_adapter_weights(model, adapter_path)

    model.to(device)
    model.eval()
    transcriptions = []

    for idx in tqdm(range(len(dataset)), desc="Generating"):
        waveform = dataset.get_waveform(idx)
        inputs = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        generated_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        transcriptions.append(transcription)

    # Sauvegarde
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line + "\n")
    print(f"📄 Transcriptions saved to {output_path}")

    return transcriptions

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--tar_path", required=True, help="Chemin vers audio_shard.tar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt")
    parser.add_argument("--n_samples", type=int, default=None, help="Nombre d'échantillons à traiter")
    args = parser.parse_args()

    dataset = TarAudioDataset(args.tar_path, n_samples=args.n_samples)
    processor, model = load_base_model(args.model_name, device=args.device)
    generate_transcriptions_from_tar(processor, model, dataset, args.tar_path, device=args.device,
                                     adapter_path=args.adapter_path, output_path=args.out)
