import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# PARAMETER
IMAGE_SIZE = (28, 28)
DATASET_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi'

def extract_geometric_features(image):
    features = []

    # Thresholding untuk mendapatkan titik-titik Braille
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Deteksi kontur (titik-titik Braille)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_count = len(contours)
    features.append(dot_count)

    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    # Jika ada titik, hitung jarak ke centroid
    if centers:
        cX_all = [p[0] for p in centers]
        cY_all = [p[1] for p in centers]
        centroid = (np.mean(cX_all), np.mean(cY_all))

        dists = [np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2) for x, y in centers]
        mean_dist = np.mean(dists)
        features.append(mean_dist)
    else:
        features.append(0)

    # Simetri horizontal
    h_sym = np.mean([abs(x - (image.shape[1] - x)) for x, _ in centers]) if centers else 0
    features.append(h_sym)

    # Simetri vertikal
    v_sym = np.mean([abs(y - (image.shape[0] - y)) for _, y in centers]) if centers else 0
    features.append(v_sym)

    return features

# Load dan proses data
data = []
labels = []

for folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            continue

        label = img_name[0].upper()
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0

        flat_feat = img.flatten()
        geom_feat = extract_geometric_features((img * 255).astype(np.uint8))
        combined_feat = np.concatenate([flat_feat, geom_feat])

        data.append(combined_feat)
        labels.append(label)

# Encode label
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Training SVM
clf = svm.SVC(kernel='rbf', gamma='scale', C=10)
clf.fit(X_train, y_train)

# Evaluasi
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Total gambar terbaca: {len(data)}")
print("Distribusi label:", Counter(labels))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
