import torch
import torchaudio
import tarfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import argparse

class AudioShardDataset:
    """Dataset pour lire les fichiers audio directement depuis un tarball"""
    def __init__(self, tar_path, processor, n_samples=None):
        self.processor = processor
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")
        members = [m for m in self.tar.getmembers() if m.isfile() and m.name.endswith(".webm")]
        if n_samples:
            members = members[:n_samples]
        self.members = members

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        member = self.members[idx]
        f = self.tar.extractfile(member)
        waveform, sr = torchaudio.load(f)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()
        input_features = self.processor.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)
        return {"input_features": input_features, "audio_name": member.name}

def load_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def load_adapter(model, adapter_path):
    adapter_state = torch.load(adapter_path, map_location="cpu")
    model_state = model.state_dict()
    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Adapter loaded from {adapter_path}")

def generate_transcriptions(dataset, processor, model, device="cuda", output_path="transcriptions.txt"):
    model.to(device)
    model.eval()
    results = []

    for item in tqdm(dataset, desc="Generating"):
        inputs = item["input_features"].unsqueeze(0).to(device)
        generated_ids = model.generate(inputs)
        transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(f"{item['audio_name']}\t{transcription.strip()}")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
    print(f"📄 Transcriptions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_shard", required=True, help="Chemin vers le tarball contenant l'audio")
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None, help="Chemin vers les poids adaptateurs")
    parser.add_argument("--n_samples", type=int, default=None, help="Nombre d'échantillons à traiter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt", help="Fichier de sortie")
    args = parser.parse_args()

    processor, model = load_model(args.model_name, device=args.device)
    if args.adapter_path:
        load_adapter(model, args.adapter_path)

    dataset = AudioShardDataset(args.audio_shard, processor, n_samples=args.n_samples)
    generate_transcriptions(dataset, processor, model, device=args.device, output_path=args.out)
import torch
import torchaudio
import tarfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import argparse

class AudioShardDataset:
    """Dataset pour lire les fichiers audio directement depuis un tarball"""
    def __init__(self, tar_path, processor, n_samples=None):
        self.processor = processor
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")
        members = [m for m in self.tar.getmembers() if m.isfile() and m.name.endswith(".webm")]
        if n_samples:
            members = members[:n_samples]
        self.members = members

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        member = self.members[idx]
        f = self.tar.extractfile(member)
        waveform, sr = torchaudio.load(f)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()
        input_features = self.processor.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)
        return {"input_features": input_features, "audio_name": member.name}

def load_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def load_adapter(model, adapter_path):
    adapter_state = torch.load(adapter_path, map_location="cpu")
    model_state = model.state_dict()
    for k, v in adapter_state.items():
        if k in model_state:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Adapter loaded from {adapter_path}")

def generate_transcriptions(dataset, processor, model, device="cuda", output_path="transcriptions.txt"):
    model.to(device)
    model.eval()
    results = []

    for item in tqdm(dataset, desc="Generating"):
        inputs = item["input_features"].unsqueeze(0).to(device)
        generated_ids = model.generate(inputs)
        transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(f"{item['audio_name']}\t{transcription.strip()}")

    with open(output_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
    print(f"📄 Transcriptions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_shard", required=True, help="Chemin vers le tarball contenant l'audio")
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None, help="Chemin vers les poids adaptateurs")
    parser.add_argument("--n_samples", type=int, default=None, help="Nombre d'échantillons à traiter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt", help="Fichier de sortie")
    args = parser.parse_args()

    processor, model = load_model(args.model_name, device=args.device)
    if args.adapter_path:
        load_adapter(model, args.adapter_path)

    dataset = AudioShardDataset(args.audio_shard, processor, n_samples=args.n_samples)
    generate_transcriptions(dataset, processor, model, device=args.device, output_path=args.out)
