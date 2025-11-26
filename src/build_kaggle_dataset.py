import os
import json
import requests
from tqdm.auto import tqdm
import argparse
from datasets import load_dataset

def build_mini_dataset(n_samples, output_dir="data"):
    # Créer dossiers
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print(f"📥 Chargement du dataset via HuggingFace (train split)...")
    dataset = load_dataset("DigitalUmuganda/ASR_Fellowship_Challenge_Dataset", split="train")

    print(f"➡️ Dataset chargé. Total samples disponibles: {len(dataset)}")

    mini_samples = []
    for i, item in enumerate(tqdm(dataset)):
        if i >= n_samples:
            break

        # Télécharger audio localement
        audio_url = item["audio"]["path"]
        audio_name = os.path.basename(audio_url)
        local_audio_path = os.path.join(audio_dir, audio_name)

        if not os.path.exists(local_audio_path):
            resp = requests.get(audio_url, stream=True)
            with open(local_audio_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        # Mettre à jour le chemin audio
        item["audio"]["path"] = os.path.join("audio", audio_name)
        mini_samples.append(item)

    # Sauvegarder metadata.jsonl
    metadata_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for sample in mini_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✅ Mini dataset prêt !")
    print(f"📁 Dossier : {output_dir}")
    print(f"📦 Nombre d'échantillons : {len(mini_samples)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Nombre d'échantillons à télécharger")
    parser.add_argument("--data_dir", default="data", help="Dossier de sortie du mini dataset")
    args = parser.parse_args()

    build_mini_dataset(args.n, args.data_dir)
