import os
import tarfile
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm
from dataset import prepare_dataset   # ta fonction existante
from utils import set_seed


# ----------------------------
# Dataset : lecture depuis un tarball
# ----------------------------
class AudioShardDataset(Dataset):
    def __init__(self, tar_path, processor, n_samples=None):
        self.samples = []
        self.processor = processor
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)

        with tarfile.open(tar_path, "r") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and m.name.endswith((".wav", ".webm"))
            ]

            if n_samples:
                members = members[:n_samples]

            for m in members:
                tar.extract(m, path=self.temp_dir)
                path = os.path.join(self.temp_dir, m.name)

                # Texte inconnu → tu ne possèdes que l’audio
                # Donc on met un label placeholder
                self.samples.append({
                    "audio_filepath": path,
                    "text": " "   # texte minimal pour fonctionner
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        waveform, sr = torchaudio.load(sample["audio_filepath"])
        # Convertir en mono si l'audio est stéréo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Whisper attend 16000Hz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Prepare_dataset crée input_features & labels
        return prepare_dataset(
            {"audio_filepath": sample["audio_filepath"], "text": sample["text"]},
            self.processor
        )


# ----------------------------
# Collate_fn: batch propre pour Whisper
# ----------------------------
def collate_fn(batch):
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_features = torch.stack(input_features)

    # Whisper veut que les labels soient -100 pour le padding
    max_len = max([l.size(0) for l in labels])
    padded_labels = torch.full((len(labels), max_len), -100)

    for i, l in enumerate(labels):
        padded_labels[i, :l.size(0)] = l

    return {
        "input_features": input_features,
        "labels": padded_labels
    }


# ----------------------------
# Chargement modèle de base
# ----------------------------
def load_base_model(model_name="openai/whisper-small", device="cuda"):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return processor, model.to(device)


# ----------------------------
# Inférence (compatible adaptateurs)
# ----------------------------
def generate_transcriptions(
    processor,
    model,
    dataloader,
    adapter_weights_path=None,
    device="cuda",
    output_path="transcriptions.txt"
):

    # Charger adaptateurs
    if adapter_weights_path:
        print(f"Loading adapters: {adapter_weights_path}")
        adapter_state = torch.load(adapter_weights_path, map_location="cpu")
        model.load_state_dict(adapter_state, strict=False)

    model.eval()
    lines = []
    refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):

            input_features = batch["input_features"].to(device)

            # >>> Correction fondamentale <<<
            # Whisper attend toujours [batch, 80, frames]
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)

            tokens = model.generate(input_features)

            transcriptions = processor.batch_decode(tokens, skip_special_tokens=True)

            for t in transcriptions:
                lines.append(t.strip())
            for ref in batch["labels"]:
                # labels → int ids. On ne peut décoder que si vrai texte disponible.
                refs.append(" ")  # placeholder

    # Sauvegarde texte
    with open(output_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    return lines, refs


# ----------------------------
# Script principal
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--tar_path", required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt")
    args = parser.parse_args()

    set_seed(42)

    processor, model = load_base_model(args.model_name, args.device)

    ds = AudioShardDataset(args.tar_path, processor, n_samples=args.n_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    generated, refs = generate_transcriptions(
        processor,
        model,
        dl,
        adapter_weights_path=args.adapter_path,
        device=args.device,
        output_path=args.out
    )

    from jiwer import wer
    WER = wer(refs, generated)
    print("WER:", WER)
    print("Évaluation terminée.")
