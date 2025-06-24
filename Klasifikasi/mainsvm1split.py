import os
import cv2
import joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# PARAMETER
IMAGE_SIZE = (28, 28)
DATASET_DIR = '/home/malik/Documents/PCD/UAS/Akuisisi'

data = []
labels = []

# Load dataset
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
        data.append(img.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Label encoding
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Grid Search
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(svm.SVC(), param_grid, cv=3, n_jobs=1, verbose=2)
grid.fit(X_train, y_train)

# Evaluasi
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)

clf = grid.best_estimator_
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print(f"Total gambar terbaca: {len(data)}")
print("Distribusi label:", Counter(labels))

# Simpan model dan label encoder
joblib.dump(clf, 'svm_model_best.pkl')
joblib.dump(le, 'label_encoder_best.pkl')
print("Model dan label encoder berhasil disimpan.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
