import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from tqdm import tqdm
import os
import torchaudio

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
    refs = []

    for item in tqdm(dataset):
        audio_path = item["audio_filepath"]
        text_ref = item.get("text", "")  # support pour dummy labels si nécessaire
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.squeeze(0).numpy()
        inputs = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        generated_tokens = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        lines.append(transcription.strip())
        refs.append(text_ref.strip())

    # Sauvegarde des transcriptions
    with open(output_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    return lines, refs

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--audio_dir", required=True, help="Chemin vers le dossier contenant les fichiers audio")
    parser.add_argument("--labels_file", default=None, help="Fichier JSON ou JSONL contenant les textes (optionnel)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="transcriptions.txt")
    args = parser.parse_args()

    # Construire le dataset local
    dataset = []
    audio_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".webm") or f.endswith(".wav")]
    audio_files.sort()  # pour reproductibilité
    labels = {}
    if args.labels_file:
        with open(args.labels_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                labels[entry["audio_filepath"]] = entry.get("text", "")

    for audio_file in audio_files:
        dataset.append({
            "audio_filepath": os.path.join(args.audio_dir, audio_file),
            "text": labels.get(audio_file, "")  # vide si pas de fichier de labels
        })

    processor, model = load_base_model(args.model_name, device=args.device)
    generated, refs = generate_transcriptions(
        processor, model, dataset, adapter_weights_path=args.adapter_path,
        device=args.device, output_path=args.out
    )

    if any(refs):
        WER = wer(refs, generated)
        print("WER:", WER)
    else:
        print("Pas de références disponibles pour calculer le WER.")
