import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm
import tarfile
import torchaudio
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from dataset import prepare_dataset
from utils import set_seed


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
        # Resampler si nécessaire
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        sample["input_values"] = waveform.squeeze().numpy()
        return prepare_dataset(sample, self.processor)

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
        for k,v in adapter_state.items():
            if k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        print("Loaded adapter weights.")

    model.to(device)
    model.eval()

    lines = []
    refs = []
    for item in tqdm(dataset):
        audio_path = item["input_features"]
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.squeeze(0).numpy()
        inputs = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        generated_tokens = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        lines.append(transcription.strip())
        refs.append(item["labels"].strip())

    with open(output_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    return lines, refs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset_name", default="DigitalUmuganda/ASR_Fellowship_Challenge_Dataset")
    parser.add_argument("--tar_path", required=True, help="Chemin vers le tarball contenant les audios")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt")
    args = parser.parse_args()

    processor, model = load_base_model(args.model_name, device=args.device)
    #ds = load_dataset(args.dataset_name)[args.split]
    ds = AudioShardDataset(args.tar_path,processor, n_samples=args.n_samples)
    generated, refs = generate_transcriptions(processor, model, ds, adapter_weights_path=args.adapter_path, device=args.device, output_path=args.out)
    from jiwer import wer
    WER = wer(refs, generated)
