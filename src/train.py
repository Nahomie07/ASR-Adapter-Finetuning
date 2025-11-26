import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW 
from hf_utils import load_whisper_model
from adapters import inject_adapters_whisper
from dataset import prepare_dataset
from utils import set_seed
from datasets import load_dataset
from tqdm import tqdm
import argparse

def freeze_base_model(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

def unfreeze_adapters(model):
    for name, p in model.named_parameters():
        if "adapter" in name or ".adapter" in name:
            p.requires_grad = True

def collate_fn(batch):
    import torch
    input_feats = [torch.tensor(b["input_features"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]
    input_feats = torch.nn.utils.rnn.pad_sequence(input_feats, batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_features": input_feats, "labels": labels}

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_whisper_model(args.model_name, device=device)

    freeze_base_model(model)

    inserted = inject_adapters_whisper(model, bottleneck_dim=args.bottleneck_dim, scale=args.scale)
    print(f"Inserted {len(inserted)} adapters.")

    unfreeze_adapters(model)

    ds = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")
    train_ds = ds["train"]

    def map_fn(x):
        return prepare_dataset(x, processor)
    train_ds = train_ds.map(lambda x: map_fn(x), remove_columns=train_ds.column_names)

    dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_parameters)
    print("Trainable params:", total_trainable)

    optimizer = AdamW(trainable_parameters, lr=args.lr)
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_training_steps)

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

    adapter_state = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            adapter_state[n] = p.detach().cpu()
    os.makedirs(args.adapter_dir, exist_ok=True)
    torch.save(adapter_state, os.path.join(args.adapter_dir, "adapter_weights.pth"))
    print("Saved adapters to", os.path.join(args.adapter_dir, "adapter_weights.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--adapter_dir", default="./adapters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
