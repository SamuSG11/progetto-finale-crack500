from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class CrackDataset:

    def __init__(self, split="train", img_size=(256,256)):

        self.root_dir = BASE_DIR / "data"

        self.split = split
        self.img_size = img_size

        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"

        self.images = sorted(list(self.images_dir.glob("*")))
        self.masks = sorted(list(self.masks_dir.glob("*")))

        print("IMAGES DIR:", self.images_dir)
        print("FOUND IMAGES:", len(self.images))
        print("FOUND MASKS:", len(self.masks))

        assert len(self.images) > 0, "NO IMAGES FOUND → path sbagliato"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        return img.astype(np.float32), mask