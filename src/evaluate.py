import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from jiwer import wer
from tqdm import tqdm
import os

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
        audio_path = item["audio_filepath"]
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
        refs.append(item["text"].strip())

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

    #ds = load_dataset(args.dataset_name)[args.split]
    ds = AudioShardDataset(args.tar_path, n_samples=args.n_samples)
    processor, model = load_base_model(args.model_name, device=args.device)
    processor, model = load_base_model(args.model_name, device=args.device)
    generated, refs = generate_transcriptions(processor, model, ds, adapter_weights_path=args.adapter_path, device=args.device, output_path=args.out)
    from jiwer import wer
    WER = wer(refs, generated)
