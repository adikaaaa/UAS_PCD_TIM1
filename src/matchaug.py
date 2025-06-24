import cv2
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk menghitung brightness dan contrast
def compute_brightness_contrast(image_gray):
    brightness = np.mean(image_gray)
    contrast = np.std(image_gray)
    return brightness, contrast

# Fungsi histogram matching
def match_histogram(source, reference):
    matched = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)

    matched[:, :, 0] = cv2.equalizeHist(matched[:, :, 0])
    ref_hist = cv2.equalizeHist(ref[:, :, 0])
    
    matched[:, :, 0] = cv2.equalizeHist(ref[:, :, 0])
    result = cv2.cvtColor(matched, cv2.COLOR_YCrCb2BGR)
    return result

# Load gambar (ubah path sesuai lokasi file Anda)
img1 = cv2.imread("/home/malik/Documents/PCD/UAS/Akuisisi/DatasetCapture/D_1.png")  # Gambar sumber
img2 = cv2.imread("/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters/D.jpg")       # Gambar referensi

# Resize jika perlu agar sama
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# Konversi ke grayscale untuk hitung brightness dan contrast
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

brightness1, contrast1 = compute_brightness_contrast(gray1)
brightness2, contrast2 = compute_brightness_contrast(gray2)

print(f"[INFO] Gambar 1 - Brightness: {brightness1:.2f}, Contrast: {contrast1:.2f}")
print(f"[INFO] Gambar 2 - Brightness: {brightness2:.2f}, Contrast: {contrast2:.2f}")

# Matching histogram gambar1 agar mirip gambar2
adjusted = match_histogram(img1, img2)

# Simpan hasil
cv2.imwrite("adjusted_result.jpg", adjusted)
print("[INFO] Gambar hasil disimpan sebagai 'adjusted_result.jpg'")
