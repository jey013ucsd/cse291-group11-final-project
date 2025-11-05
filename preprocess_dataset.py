from pathlib import Path
import pickle
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

def crop_square(img):
    w, h = img.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    return img.crop((l, t, l + s, t + s))

def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    target_dir = Path(cfg["target_dir"])
    img_size = int(cfg["IMG_SIZE"])

    for split in ["train", "test", "validation"]:
        split_dir = target_dir / split
        if not split_dir.exists():
            continue

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            files = sorted(label_dir.glob("*.jpeg"))
            if not files:
                continue

            print(f"[{split}/{label_dir.name}] processing {len(files)} images")

            for f in tqdm(files, desc=f"{split}/{label_dir.name}", ncols=80):
                with Image.open(f) as im:
                    im = im.convert("L")
                    im = crop_square(im).resize((img_size, img_size), Image.Resampling.LANCZOS)

                    # save processed png
                    png_path = label_dir / f"{f.stem}.png"
                    im.save(png_path, format="PNG", optimize=True)

                    # save pickle
                    arr = np.asarray(im, dtype=np.uint8)
                    meta = {
                        "split": split,
                        "label": label_dir.name,
                        "img_size": img_size,
                        "source": f.name,
                        "png_path": str(png_path),
                    }
                    with open(label_dir / f"{f.stem}.pkl", "wb") as pf:
                        pickle.dump({"image": arr, "meta": meta}, pf, protocol=4)

                f.unlink()

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()