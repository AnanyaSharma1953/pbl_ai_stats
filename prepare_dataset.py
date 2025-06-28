import os
import shutil
from tqdm import tqdm

# Define your dataset paths
MENDELEY_1 = "data/mendeley_1"
MENDELEY_2_AUGMENTED = "data/mendeley_2/Augmented Images (Primary Source)"
MENDELEY_2_ORIGINAL = "data/mendeley_2/Original Images (Primary and Secondary Sources)"
MENDELEY_3 = "data/mendeley_3"
COMBINED_DIR = "data/combined"

# Function to copy images from a dataset split folder
def copy_images(src_folder, dst_folder, split="train"):
    for class_name in os.listdir(src_folder):
        class_src_path = os.path.join(src_folder, class_name)
        if not os.path.isdir(class_src_path):
            continue

        class_dst_path = os.path.join(dst_folder, split, class_name)
        os.makedirs(class_dst_path, exist_ok=True)

        for img_name in tqdm(os.listdir(class_src_path), desc=f"{split} - {class_name}"):
            img_src = os.path.join(class_src_path, img_name)
            img_dst = os.path.join(class_dst_path, img_name)

            # Only copy if it's an image
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(img_src, img_dst)

# ✅ Step 1: Clear/Create combined directory
if os.path.exists(COMBINED_DIR):
    shutil.rmtree(COMBINED_DIR)
os.makedirs(COMBINED_DIR)

# ✅ Step 2: Copy mendeley_1 data into train
copy_images(MENDELEY_1, COMBINED_DIR, split="train")

# ✅ Step 3: Copy both folders from mendeley_2 into train
copy_images(MENDELEY_2_ORIGINAL, COMBINED_DIR, split="train")
copy_images(MENDELEY_2_AUGMENTED, COMBINED_DIR, split="train")

# ✅ Step 4: Copy mendeley_3 data (train, val, test)
for split in ["train", "val", "test"]:
    src_path = os.path.join(MENDELEY_3, split)
    if os.path.exists(src_path):
        copy_images(src_path, COMBINED_DIR, split=split)

print("\n✅ Dataset combined successfully into:", COMBINED_DIR)
