import cv2
import joblib
import numpy as np

# Parameter
MODEL_PATH = "svm_model92.pkl"
ENCODER_PATH = "label_encoder92.pkl"
IMAGE_SIZE = (28, 28)
roi_width, roi_height = 100, 100

# Load model dan label encoder
clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Kamera tidak tersedia.")
        break

    frame_height, frame_width = frame.shape[:2]

    # Koordinat ROI (di tengah frame)
    x1 = (frame_width - roi_width) // 2
    y1 = (frame_height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    # Ambil ROI dari frame asli
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, IMAGE_SIZE)
    normalized_roi = resized_roi / 255.0
    flat_roi = normalized_roi.flatten().reshape(1, -1)

    # Prediksi
    prediction = clf.predict(flat_roi)
    predicted_label = le.inverse_transform(prediction)[0]

    # Gambar kotak ROI dan label pada frame asli (tanpa mirror)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediksi: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Tampilkan frame asli tanpa mirror
    cv2.imshow("Identifikasi Huruf Braille (Realtime)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import joblib
# import numpy as np

# # Parameter
# MODEL_PATH = "svm_model92.pkl"
# ENCODER_PATH = "label_encoder92.pkl"
# IMAGE_SIZE = (28, 28)
# roi_width, roi_height = 200, 130

# # Load model dan label encoder
# clf = joblib.load(MODEL_PATH)
# le = joblib.load(ENCODER_PATH)

# # Buka webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("[ERROR] Kamera tidak tersedia.")
#         break

#     # Buat frame mirror hanya untuk tampilan
#     mirrored_frame = cv2.flip(frame, 1)

#     frame_height, frame_width = mirrored_frame.shape[:2]

#     # Koordinat ROI (sama untuk mirror maupun asli karena simetris)
#     x1 = (frame_width - roi_width) // 2
#     y1 = (frame_height - roi_height) // 2
#     x2 = x1 + roi_width
#     y2 = y1 + roi_height

#     # Ambil ROI dari frame ASLI untuk prediksi (bukan mirrored_frame)
#     roi = frame[y1:y2, x1:x2]  # <-- gunakan frame asli, bukan mirror
#     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     resized_roi = cv2.resize(gray_roi, IMAGE_SIZE)
#     normalized_roi = resized_roi / 255.0
#     flat_roi = normalized_roi.flatten().reshape(1, -1)

#     # Prediksi
#     prediction = clf.predict(flat_roi)
#     predicted_label = le.inverse_transform(prediction)[0]

#     # Gambar kotak dan label di mirror view
#     cv2.rectangle(mirrored_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(mirrored_frame, f"Prediksi: {predicted_label}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#     # Tampilkan frame mirror
#     cv2.imshow("Identifikasi Huruf Braille (Realtime)", mirrored_frame)

#     # Tombol keluar
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
