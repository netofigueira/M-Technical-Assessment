import os
import kagglehub

DATASET_PATH = kagglehub.dataset_download("balraj98/massachusetts-roads-dataset")

IMAGE_DIR = os.path.join(DATASET_PATH, "tiff")
TRAIN_DIR = os.path.join(IMAGE_DIR, "train")
TRAIN_LABELS_DIR = os.path.join(IMAGE_DIR, "train_labels")
VAL_DIR = os.path.join(IMAGE_DIR, "val")
VAL_LABELS_DIR = os.path.join(IMAGE_DIR, "val_labels")


def rename_tif_to_tiff(folder):
    if not os.path.exists(folder):
        return
    for fname in os.listdir(folder):
        if fname.lower().endswith('.tif'):
            old_path = os.path.join(folder, fname)
            new_path = os.path.join(folder, fname + 'f')  # .tif → .tiff
            os.rename(old_path, new_path)
            print(f"Renamed: {fname} → {os.path.basename(new_path)}")

rename_tif_to_tiff(TRAIN_LABELS_DIR)
rename_tif_to_tiff(VAL_LABELS_DIR)

PSEUDO_LABELS_DIR = os.path.join(".", "pseudo_labels")
SAVED_MODEL_PATH = os.path.join(".", "model_random_0.200.pth")
