import os
import json
import tarfile
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from hf_utils import load_whisper_model
from adapters import inject_adapters_whisper
from dataset import prepare_dataset
from utils import set_seed
from tqdm.auto import tqdm
import argparse

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

def load_local_dataset(dataset_path, processor, n_samples=None):
    """Charge le dataset depuis un fichier tar ou JSONL"""
    samples = []
    if dataset_path.endswith(".tar"):
        with tarfile.open(dataset_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    sample = json.load(f)
                    sample = prepare_dataset(sample, processor)
                    samples.append(sample)
                    if n_samples and len(samples) >= n_samples:
                        break
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                sample = prepare_dataset(sample, processor)
                samples.append(sample)
                if n_samples and len(samples) >= n_samples:
                    break
    return samples

def generate_transcriptions(model, processor, dataset, device):
    """Génère des transcriptions à partir du modèle et du dataset"""
    model.eval()
    model.to(device)
    results = []
    for sample in tqdm(dataset):
        input_feats = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model.generate(input_features=input_feats)
        transcription = processor.batch_decode(outputs)[0]
        results.append(transcription)
    return results

def save_to_file(transcriptions, path):
    with open(path, "w", encoding="utf-8") as f:
        for t in transcriptions:
            f.write(t + "\n")
    print(f"Saved to {path}")

# ------------------------
# Fonction principale
# ------------------------
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le modèle de base
    processor, model = load_whisper_model(args.model_name)
    freeze_base_model(model)

    # Générer les transcriptions du modèle de base
    print("📄 Génération des transcriptions du modèle de base...")
    test_dataset = load_local_dataset(
        os.path.join(args.data_dir, "train-00000.tar"), processor, n_samples=args.n_test
    )
    base_transcriptions = generate_transcriptions(model, processor, test_dataset, device)
    save_to_file(base_transcriptions, "base_transcriptions.txt")

    # Ajouter les adaptateurs et fine-tuning
    inserted = inject_adapters_whisper(model, bottleneck_dim=args.bottleneck_dim, scale=args.scale)
    print(f"Inserted {len(inserted)} adapters.")
    unfreeze_adapters(model)

    train_dataset = load_local_dataset(
        os.path.join(args.data_dir, "train-00000.tar"), processor, n_samples=args.n_train
    )
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_parameters))

    optimizer = AdamW(trainable_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=50, num_training_steps=len(dataloader)*args.num_epochs
    )

    model.train()
    model.to(device)

    print("🚀 Début du fine-tuning des adaptateurs...")
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

    # Sauvegarder les adaptateurs
    os.makedirs(args.adapter_dir, exist_ok=True)
    adapter_state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    adapter_path = os.path.join(args.adapter_dir, "adapter_weights.pth")
    torch.save(adapter_state, adapter_path)
    print("Saved adapters to", adapter_path)

    # Générer les transcriptions du modèle affiné
    print("📄 Génération des transcriptions du modèle affiné...")
    finetuned_transcriptions = generate_transcriptions(model, processor, test_dataset, device)
    save_to_file(finetuned_transcriptions, "finetuned_transcriptions.txt")

# ------------------------
# Entrée principale
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--data_dir", default="/content/drive/MyDrive/ASR_dataset")  # dossier où train-00000.tar se trouve
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--adapter_dir", default="./adapters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=50, help="Nombre d'échantillons pour l'entraînement")
    parser.add_argument("--n_test", type=int, default=20, help="Nombre d'échantillons pour test")
    args = parser.parse_args()
    train(args)
