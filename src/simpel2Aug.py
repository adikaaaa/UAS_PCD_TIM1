import cv2
import os
import numpy as np
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)
cv2.setRNGSeed(42)


# Parameter
IMAGE_SIZE = (28, 28)
INPUT_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters'
OUTPUT_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi/simple1_augmented'

# Pastikan output folder dibuat
os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(image):
    augmented = []

    # # Flip horizontal
    # try:
    #     flipped = cv2.flip(image, 1)
    #     augmented.append(('flip', flipped))
    # except:
    #     pass

    # Rotasi acak
    try:
        angle = random.randint(-15, 15)
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(('rotate', rotated))
    except:
        pass

    # Zoom (crop tengah lalu resize)
    try:
        zoom_factor = 0.85
        zh, zw = int(h * zoom_factor), int(w * zoom_factor)
        start_x, start_y = (w - zw) // 2, (h - zh) // 2
        cropped = image[start_y:start_y + zh, start_x:start_x + zw]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            zoomed = cv2.resize(cropped, (w, h))
            augmented.append(('zoom', zoomed))
    except:
        pass

    # Noise gaussian
    # try:
    #     noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    #     noisy = cv2.add(image, noise)
    #     augmented.append(('noise', noisy))
    # except:
    #     pass

    return augmented

# Loop semua gambar
for filename in tqdm(os.listdir(INPUT_DIR), desc="Augmenting Images"):
    if not filename.lower().endswith('.png'):
        continue

    filepath = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARNING] Gagal membaca gambar: {filepath}")
        continue

    image = cv2.resize(image, IMAGE_SIZE)

    # Simpan gambar asli
    original_path = os.path.join(OUTPUT_DIR, filename)
    if not cv2.imwrite(original_path, image):
        print(f"[ERROR] Gagal simpan gambar asli: {original_path}")
    else:
        print(f"[OK] Simpan asli: {original_path}")

    # Simpan hasil augmentasi
    base_name = os.path.splitext(filename)[0]
    for aug_type, aug_img in augment_image(image):
        aug_name = f"{base_name}_{aug_type}.png"
        aug_path = os.path.join(OUTPUT_DIR, aug_name)
        success = cv2.imwrite(aug_path, aug_img)
        if not success:
            print(f"[ERROR] Gagal simpan {aug_type}: {aug_path}")
        else:
            print(f"[OK] Simpan {aug_type}: {aug_path}")
