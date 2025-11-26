
from torch.optim import AdamW 

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import  get_linear_schedule_with_warmup
from hf_utils import load_whisper_model
from adapters import inject_adapters_whisper
from dataset import prepare_dataset
from utils import set_seed
from datasets import Dataset
from tqdm import tqdm
import argparse

def freeze_base_model(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

def unfreeze_adapters(model):
    for name, p in model.named_parameters():
        if "adapter" in name:
            p.requires_grad = True

def collate_fn(batch):
    input_feats = [torch.tensor(b["input_features"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    input_feats = torch.nn.utils.rnn.pad_sequence(
        input_feats, batch_first=True, padding_value=0.0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {"input_features": input_feats, "labels": labels}


def load_local_dataset(data_dir):
    """Charge le mini dataset local téléchargé manuellement."""
    
    meta_path = os.path.join(data_dir, "metadata.jsonl")
    audio_dir = os.path.join(data_dir, "audio")

    assert os.path.exists(meta_path), "metadata.jsonl introuvable"
    assert os.path.exists(audio_dir), "dossier audio introuvable"

    samples = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            # Ré-écrire le chemin local à la place du chemin HuggingFace
            filename = os.path.basename(s["audio"]["path"])
            s["audio"]["path"] = os.path.join(audio_dir, filename)
            samples.append(s)

    return Dataset.from_list(samples)


def train(args):

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Whisper-small + tokenizer/processor
    processor, model = load_whisper_model(args.model_name, device=device)

    # Freeze base model
    freeze_base_model(model)

    # Inject adapters
    inserted = inject_adapters_whisper(
        model, bottleneck_dim=args.bottleneck_dim, scale=args.scale
    )
    print(f"Inserted {len(inserted)} adapters.")

    # Train adapters only
    unfreeze_adapters(model)

    # ================================================================
    #               CHARGEMENT DU MINI DATASET LOCAL
    # ================================================================
    dataset = load_local_dataset(args.data_dir)
    print(f"Dataset local chargé : {len(dataset)} échantillons.")

    # ================================================================
    #                     PRÉPROCESSING (HF)
    # ================================================================
    def map_fn(x):
        return prepare_dataset(x, processor)

    dataset = dataset.map(
        map_fn,
        remove_columns=dataset.column_names,
        desc="Préparation dataset"
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Paramètres entraînables :", sum(p.numel() for p in trainable))

    optimizer = AdamW(trainable, lr=args.lr)

    num_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=num_steps
    )

    model.train()
    model.to(device)

    # ================================================================
    #                       TRAIN LOOP
    # ================================================================
    for epoch in range(args.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            input_feats = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_features=input_feats,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": loss.item()})

    # ================================================================
    #               SAVE ADAPTER WEIGHTS
    # ================================================================
    os.makedirs(args.adapter_dir, exist_ok=True)

    adapter_state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    save_path = os.path.join(args.adapter_dir, "adapter_weights.pth")
    torch.save(adapter_state, save_path)

    print("Adaptateurs enregistrés dans :", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")   # 🔥 ton dataset local
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--adapter_dir", default="/kaggle/working")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
