import os
import json
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
import argparse
import requests


REPO_ID = "DigitalUmuganda/ASR_Fellowship_Challenge_Dataset"
METADATA_FILE = "metadata.jsonl"     # <-- CORRECT !


def load_metadata_local(n):
    print(f"📥 Téléchargement du metadata.jsonl (racine du repo)…")

    metadata_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=METADATA_FILE,
        repo_type="dataset"
    )

    print("➡️  Fichier metadata récupéré :", metadata_path)

    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            samples.append(sample)
            if len(samples) >= n:
                break

    print(f"➡️  {len(samples)} entrées chargées.")
    return samples


def download_audio(samples, audio_dir):
    print("🎧 Téléchargement des fichiers audio...")

    os.makedirs(audio_dir, exist_ok=True)

    for s in tqdm(samples):
        url = s["audio"]["path"]
        filename = os.path.basename(url)
        out_path = os.path.join(audio_dir, filename)

        if os.path.exists(out_path):
            s["audio"]["path"] = out_path
            continue

        r = requests.get(url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        s["audio"]["path"] = out_path

    return samples


def save_metadata(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("📄 metadata.jsonl enregistré :", output_path)


def main(args):
    data_dir = args.data_dir
    audio_dir = os.path.join(data_dir, "audio")
    os.makedirs(data_dir, exist_ok=True)

    samples = load_metadata_local(args.n)
    samples = download_audio(samples, audio_dir)

    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    save_metadata(samples, metadata_path)

    print("✅ Mini dataset prêt !")
    print(f"📁 Dossier : {data_dir}")
    print(f"📦 Nombre d’échantillons finaux : {len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()
    main(args)
