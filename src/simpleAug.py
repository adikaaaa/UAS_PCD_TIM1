import os
import numpy as np
from PIL import Image, ImageEnhance
import random
import cv2

# Set seed untuk konsistensi hasil
random.seed(42)
np.random.seed(42)
cv2.setRNGSeed(42)

# Folder input dan output
input_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/DatasetCapture copy"
output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/simple_augmented1"
os.makedirs(output_dir, exist_ok=True)

# Ukuran gambar (jika ingin distandarisasi)
image_size = (28, 28)

# Fungsi augmentasi: setiap metode disalin 5 kali
def augment_image_all_methods(img):
    augmented = []

    for i in range(5):  # 5 salinan untuk setiap metode

        # Rotasi
        angle = random.uniform(-5, 5)
        rotated = img.rotate(angle)
        augmented.append(rotated)

        # Translasi
        dx, dy = random.randint(-3, 3), random.randint(-3, 3)
        translated = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
        augmented.append(translated)

        # Brightness
        brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        augmented.append(brightness)

        # Kontras
        contrast = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        augmented.append(contrast)

        # Blur (Gaussian)
        img_np = np.array(img)
        blurred_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)
        blurred = Image.fromarray(blurred_np)
        augmented.append(blurred)

    return augmented  # total 25 gambar (5 metode × 5)

# Proses semua gambar
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        label = filename[0].upper()
        path = os.path.join(input_dir, filename)
        original_img = Image.open(path).convert("L").resize(image_size)

        augmented_images = augment_image_all_methods(original_img)
        for idx, aug_img in enumerate(augmented_images):
            save_name = f"{label}_aug_{idx}.png"
            save_path = os.path.join(output_dir, save_name)
            aug_img.save(save_path)

print("[INFO] Augmentasi selesai. Semua gambar disimpan di:", output_dir)
