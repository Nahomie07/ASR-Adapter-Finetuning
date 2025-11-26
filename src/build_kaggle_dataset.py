import os
import json
import requests
from tqdm.auto import tqdm
import argparse

"""
====================================================
   BUILD MINI DATASET FOR KAGGLE (DOCUMENTED)
====================================================

Ce script permet de créer un jeu de données local à 
partir du dataset ASR Fellowship Challenge *sans*
télécharger les ~57 Go du dataset complet.

Il télécharge uniquement les N premières lignes du 
fichier metadata.jsonl + les fichiers audio associés.

Structure générée :
    data/
      ├── metadata.jsonl
      └── audio/
            ├── xxxx.wav
            ├── yyyy.wav

Ce dataset local est compatible avec le train.py corrigé 
et n'utilise qu'une fraction de l'espace disque Kaggle.
"""


def download_metadata(index_url, n):
    """
    Télécharge uniquement les N premières lignes du metadata.jsonl
    depuis HuggingFace, sans télécharger tout le dataset.
    """
    print(f"📥 Téléchargement des {n} entrées metadata...")

    samples = []
    with requests.get(index_url, stream=True) as r:
        for line in r.iter_lines():
            if line:
                sample = json.loads(line)
                samples.append(sample)

                if len(samples) >= n:
                    break
    return samples


def download_audio(samples, audio_dir):
    """
    Télécharge les fichiers audio correspondant aux échantillons.
    """
    print("🎧 Téléchargement des fichiers audio...")

    os.makedirs(audio_dir, exist_ok=True)

    for s in tqdm(samples):
        url = s["audio"]["path"]
        local_name = os.path.basename(url)
        out_path = os.path.join(audio_dir, local_name)

        # Skip si déjà téléchargé
        if os.path.exists(out_path):
            continue

        r = requests.get(url, stream=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Remplace le chemin HF par le chemin local
        s["audio"]["path"] = out_path

    return samples


def save_metadata(samples, output_path):
    """
    Enregistre les métadonnées finales (avec chemins audio locaux).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("📄 metadata.jsonl enregistré :", output_path)


def main(args):
    # URLs officielles du dataset DigitalUmuganda
    index_url = (
        "https://huggingface.co/datasets/DigitalUmuganda/"
        "ASR_Fellowship_Challenge_Dataset/resolve/main/data/train/metadata.jsonl"
    )

    data_dir = args.data_dir
    audio_dir = os.path.join(data_dir, "audio")
    os.makedirs(data_dir, exist_ok=True)

    # Étape 1 : Télécharger N lignes du metadata.jsonl
    samples = download_metadata(index_url, args.n)

    # Étape 2 : Télécharger seulement les audios correspondants
    samples = download_audio(samples, audio_dir)

    # Étape 3 : Sauvegarder metadata.jsonl local
    metadata_path = os.path.join(data_dir, "metadata.jsonl")
    save_metadata(samples, metadata_path)

    print("✅ Mini dataset complet et prêt pour l'entraînement !")
    print(f"📁 Dossier : {data_dir}")
    print(f"📦 Nombre d'échantillons : {len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300,
                        help="Nombre d'échantillons à télécharger (défaut=300)")
    parser.add_argument("--data_dir", default="data",
                        help="Dossier de sortie du dataset local")
    args = parser.parse_args()
    main(args)
