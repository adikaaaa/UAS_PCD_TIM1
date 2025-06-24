import cv2
import os

# Parameter
SAVE_DIR = "/home/malik/Documents/PCD/UAS/Akuisisi/DatasetCapture"
os.makedirs(SAVE_DIR, exist_ok=True)

# Ukuran ROI
roi_width, roi_height = 200, 130

# Input label
label = input("Masukkan label huruf (misal: A): ").strip().upper()
existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(label + "_")]
counter = len(existing_files) + 1

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membuka kamera.")
        break

    # Ukuran frame
    frame_height, frame_width = frame.shape[:2]

    # Hitung koordinat ROI
    x1 = (frame_width - roi_width) // 2
    y1 = (frame_height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height

    # Buat mirror untuk tampilan saja
    mirrored_display = cv2.flip(frame, 1)
    display_frame = mirrored_display.copy()

    # Tambahkan kotak dan teks
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_frame, f"Label: {label} | 'c'=capture | 'q'=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tampilkan tampilan mirror
    cv2.imshow("Live Webcam (Mirror View)", display_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Simpan gambar dari frame asli (tidak mirror)
        roi = frame[y1:y2, x1:x2]
        save_path = os.path.join(SAVE_DIR, f"{label}_{counter}.png")
        cv2.imwrite(save_path, roi)
        print(f"[INFO] Gambar disimpan (tidak mirror): {save_path}")
        counter += 1

cap.release()
cv2.destroyAllWindows()
