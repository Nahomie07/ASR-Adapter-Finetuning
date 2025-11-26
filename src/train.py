import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from hf_utils import load_whisper_model
from adapters import inject_adapters_whisper
from dataset import prepare_dataset
from utils import set_seed
from torch.optim import AdamW 
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse

class StreamingASRDataset(IterableDataset):
    def __init__(self, split="train"):
        self.ds = load_dataset(
            "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset",
            split=split,
            streaming=True
        )

    def __iter__(self):
        for sample in self.ds:
            yield sample


def collate_fn(batch):
    import torch
    input_feats = [torch.tensor(b["input_features"]) for b in batch]
    labels = [torch.tensor(b["labels"]) for b in batch]

    input_feats = torch.nn.utils.rnn.pad_sequence(
        input_feats, batch_first=True, padding_value=0.0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    return {"input_features": input_feats, "labels": labels}


def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_whisper_model(args.model_name, device=device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Inject adapters
    inject_adapters_whisper(model, bottleneck_dim=args.bottleneck_dim, scale=args.scale)

    # Unfreeze adapters only
    for n, p in model.named_parameters():
        if "adapter" in n:
            p.requires_grad = True

    # STREAMING dataset
    raw_dataset = StreamingASRDataset(split="train")

    # Preprocessing by map is not possible with streaming → do it manually in collate loop

    def preprocess_generator():
        for sample in raw_dataset:
            try:
                processed = prepare_dataset(sample, processor)
                yield processed
            except Exception as e:
                print("Erreur préprocessing:", e)

    train_iter = preprocess_generator()

    dataloader = DataLoader(
        train_iter,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    # Optimizer on adapter params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    # Training loop (streaming)
    model.train()
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}")
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(input_features=batch["input_features"], labels=batch["labels"])
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print(f"Step {step} - Loss: {loss.item():.4f}")

            # stop after limited number of steps per epoch (streaming=inf)
            if step >= args.max_steps_per_epoch:
                break

    # Save adapter weights
    torch.save(
        {k: v.cpu() for k, v in model.state_dict().items() if "adapter" in k},
        f"{args.adapter_dir}/adapter_weights.pth"
    )
    print("Adapters saved !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--max_steps_per_epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--adapter_dir", default="./adapters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)

