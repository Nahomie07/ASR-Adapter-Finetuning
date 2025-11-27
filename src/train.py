import os
import tarfile
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from hf_utils import load_whisper_model
from adapters import inject_adapters_whisper
from dataset import prepare_dataset
from utils import set_seed
from tqdm.auto import tqdm
import argparse
import torchaudio

# ------------------------
# Dataset à partir d'un tarball
# ------------------------
class AudioShardDataset(Dataset):
    def __init__(self, tar_path, processor, n_samples=None):
        self.samples = []
        self.processor = processor
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Extraire les fichiers du tarball
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith((".webm", ".wav"))]
            if n_samples:
                members = members[:n_samples]
            for m in members:
                tar.extract(m, path=self.temp_dir)
                audio_path = os.path.join(self.temp_dir, m.name)
                self.samples.append({"audio_filepath": audio_path, "text": "dummy text"})  # texte dummy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Charger l'audio
        waveform, sr = torchaudio.load(sample["audio_filepath"])
        # Convertir en mono si l'audio est stéréo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resampler si nécessaire
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        sample["input_values"] = waveform.squeeze().numpy()
        return prepare_dataset(sample, self.processor)

# ------------------------
# Fonctions utilitaires
# ------------------------
def freeze_base_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_adapters(model):
    for name, p in model.named_parameters():
        if "adapter" in name or ".adapter" in name:
            p.requires_grad = True

def collate_fn(batch):
    input_feats = [torch.tensor(b["input_features"], dtype=torch.float32) for b in batch]
    labels = [torch.tensor(b["labels"], dtype=torch.long) for b in batch]
    input_feats = torch.nn.utils.rnn.pad_sequence(input_feats, batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_feats, "labels": labels}

# ------------------------
# Entraînement
# ------------------------
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger modèle
    processor, model = load_whisper_model(args.model_name)
    freeze_base_model(model)

    # Ajouter adaptateurs
    inserted = inject_adapters_whisper(model, bottleneck_dim=args.bottleneck_dim, scale=args.scale)
    print(f"Inserted {len(inserted)} adapters.")
    unfreeze_adapters(model)

    # Dataset
    dataset = AudioShardDataset(args.audio_shard, processor, n_samples=args.n_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(dataloader)*args.num_epochs
    )

    model.train()
    model.to(device)

    for epoch in range(args.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            input_feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_features=input_feats, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})

    # Sauvegarder adaptateurs
    os.makedirs(args.adapter_dir, exist_ok=True)
    adapter_state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(adapter_state, os.path.join(args.adapter_dir, "adapter_weights.pth"))
    print("Saved adapters to", os.path.join(args.adapter_dir, "adapter_weights.pth"))

# ------------------------
# Entrée principale
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--audio_shard", required=True, help="Chemin vers le tarball audio")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--adapter_dir", default="./adapters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=20)
    args = parser.parse_args()
    train(args)
