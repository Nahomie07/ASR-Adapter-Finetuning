import os
import tarfile
import json
import requests
from tqdm.auto import tqdm
import argparse
import shutil

# URL du premier tar d'entraînement
TAR_URLS = [
    "https://huggingface.co/datasets/DigitalUmuganda/ASR_Fellowship_Challenge_Dataset/resolve/main/train_tarred/train-00000.tar",
    # Tu peux ajouter d'autres tar si nécessaire
]

def download_tar(tar_url, local_path):
    if os.path.exists(local_path):
        print(f"✅ {local_path} déjà téléchargé")
        return
    print(f"📥 Téléchargement de {tar_url} ...")
    r = requests.get(tar_url, stream=True)
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    print(f"✅ {local_path} téléchargé")


def extract_samples(tar_path, n_samples, output_dir):
    print(f"📦 Extraction de {n_samples} échantillons depuis {tar_path} ...")
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    samples = []
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        count = 0
        i = 0
        while count < n_samples and i < len(members):
            member = members[i]
            i += 1
            # Chaque échantillon a 2 fichiers : audio.wav + metadata.json
            if member.name.endswith(".json"):
                f = tar.extractfile(member)
                if f:
                    meta = json.load(f)
                    # Déplacer l'audio correspondant
                    audio_name = meta["audio"]["path"].split("/")[-1]
                    audio_member = next((m for m in members if m.name.endswith(audio_name)), None)
                    if audio_member:
                        tar.extract(audio_member, path=audio_dir)
                        meta["audio"]["path"] = os.path.join("audio", audio_name)
                    samples.append(meta)
                    count += 1
    return samples


def save_metadata(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("📄 metadata.jsonl sauvegardé :", output_path)


def main(args):
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    all_samples = []
    for idx, tar_url in enumerate(TAR_URLS):
        tar_local = os.path.join(data_dir, f"train-{idx:05d}.tar")
        download_tar(tar_url, tar_local)

        remaining = args.n - len(all_samples)
        if remaining <= 0:
            break

        samples = extract_samples(tar_local, remaining, data_dir)
        all_samples.extend(samples)

    save_metadata(all_samples, os.path.join(data_dir, "metadata.jsonl"))
    print("✅ Mini dataset prêt !")
    print(f"📁 Dossier : {data_dir}")
    print(f"📦 Nombre d’échantillons : {len(all_samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Nombre d'échantillons à extraire")
    parser.add_argument("--data_dir", default="data", help="Dossier de sortie du mini dataset")
    args = parser.parse_args()
    main(args)
