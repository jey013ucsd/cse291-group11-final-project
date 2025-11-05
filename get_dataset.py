import os
import yaml
from datasets import load_dataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
DATASET_NAME = config["dataset_name"]
TARGET_DIR = config["target_dir"]

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    print(f"Downloading {DATASET_NAME} into {TARGET_DIR}/ ...")
    ds_dict = load_dataset(DATASET_NAME)

    for split_name, split in ds_dict.items():
        split_dir = os.path.join(TARGET_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i, example in enumerate(split):
            img = example["image"]
            label = example["label"]
            label_name = split.features["label"].int2str(label)

            label_dir = os.path.join(split_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)

            file_path = os.path.join(label_dir, f"{split_name}_{i:05d}.jpeg")
            img.save(file_path)

            if (i + 1) % 500 == 0:
                print(f"  Saved {i + 1} images in '{split_name}'")

    print("Download complete.")

if __name__ == "__main__":
    main()