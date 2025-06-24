import os
import cv2
import numpy as np
from collections import Counter

# PARAMETER
IMAGE_SIZE = (28, 28)

# Daftar folder dataset eksplisit (ganti sesuai lokasi masing-masing)
DATASET_DIRS = [
    '/home/malik/Documents/PCD/UAS/Akuisisi/braille_augmented',
    '/home/malik/Documents/PCD/UAS/Akuisisi/3ofkind_augmented',
    '/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters'  # kalau ada
]

data = []
labels = []

# Loop setiap direktori dataset
for dataset_path in DATASET_DIRS:
    if not os.path.isdir(dataset_path):
        print(f"[PERINGATAN] Folder tidak ditemukan: {dataset_path}")
        continue

    # Loop semua file gambar di folder tersebut
    for img_name in sorted(os.listdir(dataset_path)):
        img_name = img_name.strip()

        if not img_name.lower().endswith('.png'):
            continue

        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[ERROR] Gagal membaca gambar: {img_path}")
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0
        data.append(img.flatten())

        label = img_name.split('_')[0].upper()
        labels.append(label)

# Ringkasan
print(f"Total gambar terbaca: {len(data)}")
print("Distribusi label:", Counter(labels))
