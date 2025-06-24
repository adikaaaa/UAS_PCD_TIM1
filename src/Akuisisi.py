from PIL import Image
import os

# Path ke gambar dan folder output
image_path = "/home/malik/Documents/PCD/UAS/Akuisisi/AdobeStock_394223156-1200x675.jpeg"
output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"
os.makedirs(output_dir, exist_ok=True)

# Buka gambar
image = Image.open(image_path)

# Label huruf A-Z
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Informasi per baris: (y_start, y_end, x_start, x_end, jumlah_huruf)
baris_info = [
    (86, 150, 99, 1099, 10),   # A–J
    (255, 350, 99, 1099, 10), # K–T
    (430, 525, 295, 895, 6),  # U–Z
]

# Potong dan simpan gambar per huruf
huruf_index = 0
for baris, (y_start, y_end, x_start, x_end, jumlah_huruf) in enumerate(baris_info):
    cell_width = (x_end - x_start) // jumlah_huruf
    cell_height = y_end - y_start

    for col in range(jumlah_huruf):
        if huruf_index >= len(labels):
            break
        left = x_start + col * cell_width
        upper = y_start
        right = left + cell_width
        lower = upper + cell_height

        cropped = image.crop((left, upper, right, lower))
        filename = f"{labels[huruf_index]}.jpg"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)
        print(f"Disimpan: {filepath}")
        huruf_index += 1

print("✅ Semua huruf Braille A–Z berhasil dipotong dengan batas khusus per baris.")

# from PIL import Image
# import os

# # --- Konfigurasi awal ---
# # Ganti path ini sesuai lokasi gambar kamu
# image_path = "/home/malik/Documents/PCD/UAS/Akuisisi/AdobeStock_394223156-1200x675.jpeg"

# # Nama folder untuk menyimpan hasil potongan
# output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"

# # Buat folder jika belum ada
# os.makedirs(output_dir, exist_ok=True)

# # Buka gambar
# image = Image.open(image_path)

# # Tentukan jumlah baris dan kolom dalam gambar
# cols = 20  # 10 huruf per baris (A-J, K-T, U-Z)
# rows = 3   # 3 baris
# cell_width = image.width // cols
# cell_height = image.height // rows

# # Daftar label huruf A-Z
# labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# # Lakukan pemotongan gambar berdasarkan grid
# index = 0
# for row in range(rows):
#     for col in range(cols):
#         if index >= len(labels):
#             break  # Hentikan jika sudah 26 huruf
#         left = col * cell_width
#         upper = row * cell_height
#         right = left + cell_width
#         lower = upper + cell_height
#         cropped = image.crop((left, upper, right, lower))

#         filename = f"{labels[index]}.jpg"
#         filepath = os.path.join(output_dir, filename)
#         cropped.save(filepath)
#         print(f"Disimpan: {filepath}")
#         index += 1

# print("Pemotongan selesai. Semua huruf A-Z telah disimpan.")

# from PIL import Image
# import os

# # Path gambar
# image_path = "/home/malik/Documents/PCD/UAS/Akuisisi/AdobeStock_394223156-1200x675.jpeg"
# output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"
# os.makedirs(output_dir, exist_ok=True)

# # Buka gambar
# image = Image.open(image_path)

# # Ukuran gambar
# width, height = image.size

# # Tentukan tinggi baris
# row_height = height // 3

# # Label huruf A-Z
# labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# # Map (baris, kolom, total_kolom_pada_baris) untuk tiap huruf
# grid_map = []

# # Baris 1 & 2: masing-masing 10 kolom
# for row in range(2):
#     for col in range(10):
#         grid_map.append((row, col, 10))

# # Baris 3: hanya 6 kolom
# for col in range(6):
#     grid_map.append((2, col, 6))

# # Potong dan simpan huruf
# for idx, label in enumerate(labels):
#     row, col, total_cols = grid_map[idx]
#     cell_width = width // total_cols
#     top = row * row_height
#     bottom = top + row_height
#     left = col * cell_width
#     right = left + cell_width

#     cropped = image.crop((left, top, right, bottom))
#     filepath = os.path.join(output_dir, f"{label}.jpg")
#     cropped.save(filepath)
#     print(f"Disimpan: {filepath}")

# print("Semua huruf Braille berhasil dipotong dan disimpan.")

# from PIL import Image
# import os

# # Path ke gambar dan folder output
# image_path = "/home/malik/Documents/PCD/UAS/Akuisisi/AdobeStock_394223156-1200x675.jpeg"
# output_dir = "/home/malik/Documents/PCD/UAS/Akuisisi/braille_letters"
# os.makedirs(output_dir, exist_ok=True)

# # Buka gambar
# image = Image.open(image_path)

# # Ukuran gambar
# full_width, full_height = image.size

# # Batasi pembacaan sumbu X
# x_start = 100
# x_end = 1100
# usable_width = x_end - x_start

# # Bagi tinggi menjadi 3 baris
# row_height = full_height // 3

# # Label huruf A-Z
# labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# # Map: (row, col, total_cols_in_row)
# grid_map = []

# # Baris 1 & 2: 10 huruf
# for row in range(2):
#     for col in range(10):
#         grid_map.append((row, col, 10))

# # Baris 3: 6 huruf
# for col in range(6):
#     grid_map.append((2, col, 6))

# # Proses potong dan simpan
# for idx, label in enumerate(labels):
#     row, col, total_cols = grid_map[idx]
    
#     cell_width = usable_width // total_cols
#     top = row * row_height
#     bottom = top + row_height
#     left = x_start + col * cell_width
#     right = left + cell_width

#     cropped = image.crop((left, top, right, bottom))
#     filepath = os.path.join(output_dir, f"{label}.jpg")
#     cropped.save(filepath)
#     print(f"Disimpan: {filepath}")

# print("✅ Semua huruf Braille berhasil dipotong dan disimpan.")
