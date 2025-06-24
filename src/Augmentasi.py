import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

random.seed(42)
np.random.seed(42)
cv2.setRNGSeed(42)

# Folder input dan output
input_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"
output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_augmented"
os.makedirs(output_dir, exist_ok=True)

# Konfigurasi augmentasi
augment_per_image = 10  # banyaknya salinan tiap huruf
image_size = (28, 28)

# Fungsi augmentasi sederhana
def augment_image(img):
    # Pilih operasi acak
    ops = []

    # Rotasi ±15 derajat
    angle = random.uniform(-15, 15)
    ops.append(img.rotate(angle))

    # Translasi: geser sedikit
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    ops.append(img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy)))

    # Zoom in (crop dan resize)
    crop_margin = 4
    w, h = img.size
    cropped = img.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin)).resize(img.size)
    ops.append(cropped)

    # Tambahkan noise (dalam bentuk brightness acak)
    enhancer = ImageEnhance.Brightness(img)
    brightness = random.uniform(0.8, 1.2)
    ops.append(enhancer.enhance(brightness))

    # Inversi (opsional, kalau ingin variasi lebih ekstrem)
    inverted = Image.fromarray(255 - np.array(img))
    ops.append(inverted)

    return random.sample(ops, k=min(3, len(ops)))  # ambil 3 augmentasi dari yang tersedia

# Proses semua gambar
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        base_name = filename[0]  # Misal A.jpg → 'A'
        path = os.path.join(input_dir, filename)
        original_img = Image.open(path).convert("L").resize(image_size)

        for i in range(augment_per_image):
            augmented_images = augment_image(original_img)
            for j, aug_img in enumerate(augmented_images):
                save_name = f"{base_name}_{i}.png"
                save_path = os.path.join(output_dir, save_name)
                aug_img.save(save_path)
