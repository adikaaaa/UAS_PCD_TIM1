import os

def print_tree(root_dir, prefix="", level=0, max_level=3):
    if level > max_level:
        return
    try:
        entries = sorted(os.listdir(root_dir))
    except Exception as e:
        print(f"{prefix}[ERROR] Tidak bisa membuka {root_dir}: {e}")
        return

    for entry in entries:
        path = os.path.join(root_dir, entry)
        print(f"{prefix}|-- {entry}")
        if os.path.isdir(path):
            print_tree(path, prefix + "|   ", level + 1, max_level)

root_folder = '/home/malik/Documents/PCD/UAS/Akuisisi'
print(f"Struktur folder dari: {root_folder}")
print_tree(root_folder)
