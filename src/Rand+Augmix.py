import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2

random.seed(42)
np.random.seed(42)
cv2.setRNGSeed(42)


# === AugMix ===
class AugMix:
    def __init__(self, severity=3, width=3, alpha=1.0):
        self.severity = severity
        self.width = width
        self.alpha = alpha
        self.augmentations = [
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.1, 0.1)),
            T.GaussianBlur(kernel_size=3),
            T.RandomPerspective(distortion_scale=0.1, p=0.5),
        ]

    def __call__(self, image):
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = torch.zeros_like(F.to_tensor(image))
        for i in range(self.width):
            aug_image = image.copy()
            aug = random.choice(self.augmentations)
            aug_image = aug(aug_image)
            mix += ws[i] * F.to_tensor(aug_image)

        mixed = (1 - m) * F.to_tensor(image) + m * mix
        return F.to_pil_image(torch.clamp(mixed, 0.0, 1.0))

# === Gabungan RandAugment + AugMix ===
class CombinedAugmentation:
    def __init__(self):
        self.rand_aug = T.RandAugment(num_ops=2, magnitude=5)
        self.augmix = AugMix()

    def __call__(self, img):
        img = self.rand_aug(img)
        img = self.augmix(img)
        return img

# === Fungsi utama ===
def augment_all_images(input_dir, output_dir, n_augments=5):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    augmenter = CombinedAugmentation()
    supported_ext = ['.jpg', '.jpeg', '.png']

    for file in input_path.iterdir():
        if file.suffix.lower() not in supported_ext:
            continue

        try:
            img = Image.open(file).convert("RGB")
            base_name = file.stem  # Misal "A" dari "A.jpg"

            for i in range(1, n_augments + 1):
                aug_img = augmenter(img)
                save_name = f"{base_name}_{i}.png"
                save_path = output_path / save_name
                aug_img.save(save_path)
                print(f"[✓] Saved: {save_path}")

        except Exception as e:
            print(f"[!] Error processing {file.name}: {e}")

# === Eksekusi ===
if __name__ == "__main__":
    input_folder = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"         # Ganti jika perlu
    output_folder = "/home/malik/Documents/PCD/UAS/Akuisisi/RandAug_brailleDattaset"      # Folder hasil augmentasi
    jumlah_augmentasi_per_gambar = 5         # Ganti jumlah augmentasi per gambar

    augment_all_images(input_folder, output_folder, jumlah_augmentasi_per_gambar)
