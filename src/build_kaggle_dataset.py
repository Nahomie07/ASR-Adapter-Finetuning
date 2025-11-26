import os
import json
import requests
from tqdm.auto import tqdm
import argparse

"""
====================================================
   BUILD MINI DATASET FOR KAGGLE (FINAL + WORKING)
====================================================

Télécharge seulement N échantillons du dataset
Digital Umuganda ASR Fellowship Challenge sans
récupérer les 57 Go complets.

Fonctionne dans Kaggle.
"""


# ----------------------------------------------------------- #
#     URL DIRECTE QUI FONCTIONNE (non-LFS, non-HTML)          #
# ----------------------------------------------------------- #
METADATA_URL = (
    "https://huggingface.co/datasets/DigitalUmuganda/"
    "ASR_Fellowship_Challenge_Dataset/resolve/main/data/train/metadata.jsonl?download=1"
)


def download_metadata(n):
    print(f"📥 Téléchargement des {n} entrées metadata...")

    samples = []
    r = requests.get(METADATA_URL, stream=True)

    for raw in r.iter_lines(decode_unicode=True):
        if not raw:
            continue

        # On ignore tout ce qui n’est pas JSON
        if not raw.strip().startswith("{"):
            continue

        try:
            sample = json.loads(raw)
        except json.JSONDecodeError:
            continue

        samples.append(sample)
        if len(samples) >= n:
            break

    print(f"➡️  {len(samples)} entrées valides récupérées.")
    return samples


def download_audio(samples, audio_dir):
    print("🎧 Téléchargement des fichiers audio...")

    os.makedirs(audio_dir, exist_ok=True)

    for s in tqdm(samples):
        url = s["audio"]["path"]
        local_name = os.path.basename(url)
        out_path = os.path.join(audio_dir, local_name)

        if os.path.exists(out_path):
            s["audio"]["path"] = out_path
            continue

        r = requests.get(url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
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

    # 1) Télécharger N lignes du metadata.jsonl
    samples = download_metadata(args.n)

    # 2) Télécharger les audios
    samples = download_audio(samples, audio_dir)

    # 3) Sauvegarder metadata.jsonl local
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    save_metadata(samples, metadata_path)

    print("✅ Mini dataset complet et prêt pour l'entraînement !")
    print(f"📁 Dossier : {data_dir}")
    print(f"📦 Nombre d'échantillons finaux : {len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300,
                        help="Nombre d'échantillons à télécharger")
    parser.add_argument("--data_dir", default="data",
                        help="Dossier de sortie du dataset local")
    args = parser.parse_args()
    main(args)
