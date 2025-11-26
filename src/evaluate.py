import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm
import tarfile
import torchaudio
import argparse
import os

class AudioShardDataset:
    """Dataset pour lire un tarball d'audio avec labels fictifs (ou réels si disponibles)."""
    def __init__(self, tar_path, n_samples=None):
        self.samples = []
        self.tar_path = tar_path
        self.tar = tarfile.open(tar_path, "r")
        members = [m for m in self.tar.getmembers() if m.isfile() and m.name.endswith(".webm")]
        if n_samples:
            members = members[:n_samples]
        for m in members:
            self.samples.append(m.name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        member = self.samples[idx]
        f = self.tar.extractfile(member)
        # Sauvegarde temporaire dans RAM
        waveform, sr = torchaudio.load(f)
        return {"waveform": waveform.squeeze(0), "sr": sr, "audio_name": member}

def load_base_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model

def generate_transcriptions(processor, model, dataset, adapter_weights_path=None, device="cuda", output_path="transcriptions.txt"):
    if adapter_weights_path:
        adapter_state = torch.load(adapter_weights_path, map_location="cpu")
        model_state = model.state_dict()
        for k, v in adapter_state.items():
            if k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        print("Loaded adapter weights.")

    model.to(device)
    model.eval()

    lines = []
    refs = []  # Ici tu peux mettre les vraies transcriptions si disponibles
    for item in tqdm(dataset):
        wav, sr = item["waveform"], item["sr"]
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)
        inputs = processor.feature_extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        generated_tokens = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        lines.append(transcription.strip())

        # Si tu n’as pas les références exactes, mets une valeur vide ou dummy
        refs.append("")  

    # Sauvegarde des transcriptions
    with open(output_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    # Calcul WER si refs sont disponibles
    WER = wer(refs, lines) if all(refs) else None
    return lines, refs, WER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--tar_path", required=True, help="Chemin vers le tarball contenant les audios")
    parser.add_argument("--out", default="transcriptions.txt")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = AudioShardDataset(args.tar_path, n_samples=args.n_samples)
    processor, model = load_base_model(args.model_name, device=args.device)
    generated, refs, WER = generate_transcriptions(processor, model, dataset, adapter_weights_path=args.adapter_path, device=args.device, output_path=args.out)
    
    if WER is not None:
        print("WER:", WER)
    else:
        print("References not available, WER not computed.")
