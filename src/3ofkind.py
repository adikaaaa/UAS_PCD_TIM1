import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
cv2.setRNGSeed(42)

# Atur path input dan output
input_folder = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"         # Ganti dengan path folder gambar asli
output_folder = "/home/malik/Documents/PCD/UAS/Akuisisi/3ofkind_augmented"      # Ganti sesuai kebutuhan
os.makedirs(output_folder, exist_ok=True)

# Parameter: jumlah augmentasi per metode
n_augment = 10

# RandAugment-like: kombinasi 2 transformasi acak dari daftar
def randaugment_pipeline(image):
    transform_ops = [
        lambda x: x.rotate(random.randint(-20, 20)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.5, 1.5)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.5, 1.5)),
        lambda x: ImageEnhance.Sharpness(x).enhance(random.uniform(0.5, 2.0)),
        # lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    ]
    augmented = []
    for _ in range(n_augment // 2):
        img_aug = image.copy()
        ops = random.sample(transform_ops, 2)
        for op in ops:
            img_aug = op(img_aug)
        augmented.append(img_aug)
    return augmented

# AugMix-like: mencampur hasil dari beberapa transformasi ringan
def augmix_pipeline(image):
    transform_ops = [
        lambda x: x.rotate(random.randint(-15, 15)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8))),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
    ]
    augmented = []
    for _ in range(n_augment // 3):
        img_aug = image.copy()
        img1 = transform_ops[random.randint(0, 3)](img_aug)
        img2 = transform_ops[random.randint(0, 3)](img_aug)
        img3 = transform_ops[random.randint(0, 3)](img_aug)
        mixed = Image.blend(Image.blend(img1, img2, alpha=0.5), img3, alpha=0.5)
        augmented.append(mixed)
    return augmented

# Augmentasi klasik: flipping, noise, resizing, padding
def classic_pipeline(image):
    augmented = []
    img_np = np.array(image)

    # flipped = cv2.flip(img_np, 1)
    noisy = img_np + np.random.normal(0, 10, img_np.shape).astype(np.uint8)
    resized = cv2.resize(img_np, (int(img_np.shape[1]*0.9), int(img_np.shape[0]*0.9)))
    padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # augmented.append(Image.fromarray(flipped))
    augmented.append(Image.fromarray(noisy.clip(0, 255).astype(np.uint8)))
    augmented.append(Image.fromarray(padded))
    return augmented

# Loop utama: proses semua gambar
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(filename)[0]

        augmented_images = []
        augmented_images += randaugment_pipeline(img)
        augmented_images += augmix_pipeline(img)
        augmented_images += classic_pipeline(img)

        for idx, aug_img in enumerate(augmented_images):
            aug_name = f"{base_name}_{idx+1}.png"
            aug_img.save(os.path.join(output_folder, aug_name))
