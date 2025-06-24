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
INPUT_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters'  # Ganti sesuai folder aslinya
OUTPUT_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi/augmented_full'  # Hasil disimpan di sini

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fungsi augmentasi
def augment_image(image):
    augmented = []

    # # 1. Flip Horizontal
    # flipped = cv2.flip(image, 1)
    # augmented.append(('flip', flipped))

    # 2. Rotasi Acak
    angle = random.randint(-15, 15)
    h, w = image.shape
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    rotated = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(('rotate', rotated))

    # 3. Zoom (crop tengah lalu resize)
    zoom_factor = 0.85
    zh, zw = int(h * zoom_factor), int(w * zoom_factor)
    start_x, start_y = (w - zw) // 2, (h - zh) // 2
    cropped = image[start_y:start_y+zh, start_x:start_x+zw]
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        zoomed = cv2.resize(cropped, (w, h))
        augmented.append(('zoom', zoomed))

    # # 4. Gaussian Noise
    # noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    # noisy = cv2.add(image, noise)
    # augmented.append(('noise', noisy))

    # 5. Brightness
    bright = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    augmented.append(('bright', bright))

    # 6. Contrast
    mean = np.mean(image)
    contrast = np.clip((image - mean) * 1.5 + mean, 0, 255).astype(np.uint8)
    augmented.append(('contrast', contrast))

    # 7. Translate
    M_trans = np.float32([[1, 0, 2], [0, 1, 2]])
    translated = cv2.warpAffine(image, M_trans, (w, h), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(('translate', translated))

    # 8. Blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented.append(('blur', blurred))

    return augmented

# Loop semua gambar dan augment
for filename in tqdm(os.listdir(INPUT_DIR), desc="Augmenting Images"):
    if not filename.lower().endswith('.jpg'):
        continue

    filepath = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARNING] Gagal membaca gambar: {filepath}")
        continue

    # Resize jika belum sesuai
    if image.shape != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE)

    # Simpan original ke output
    output_original_path = os.path.join(OUTPUT_DIR, filename)
    if not cv2.imwrite(output_original_path, image):
        print(f"[ERROR] Gagal menyimpan gambar asli: {output_original_path}")

    # Augment dan simpan hasilnya
    augments = augment_image(image)
    base_name = os.path.splitext(filename)[0]

    for aug_type, aug_img in augments:
        out_name = f"{base_name}_{aug_type}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if not cv2.imwrite(out_path, aug_img):
            print(f"[ERROR] Gagal menyimpan gambar augmentasi: {out_path}")
