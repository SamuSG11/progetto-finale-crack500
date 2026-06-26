import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import base64
import random

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Crack500LocalDataLoader:
    """
    Carica il dataset CRACK500 scaricato localmente in una cartella personalizzata.
    """
    def __init__(self, batch_size=16, img_size=(256, 256)):
        self.root_dir = BASE_DIR / "data" 
        self.batch_size = batch_size
        self.img_size = img_size

    def _load_with_opencv(self, img_path, mask_path):

        # Trasformiamo i tensori di stringhe in stringhe Python classiche per cv2
        img_path_str = img_path.numpy().decode('utf-8')
        mask_path_str = mask_path.numpy().decode('utf-8')

        img = cv2.imread(img_path_str)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
  
        mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        return img.astype(np.float32), mask

    def _tf_wrapper(self, img_path, mask_path):
        """Dice a TensorFlow di poter usare la funzione OpenCV senza rompersi."""
        img, mask = tf.py_function(
            func=self._load_with_opencv,
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )
        # Forziamo le dimensioni statiche (altrimenti Keras si lamenta)
        img.set_shape([self.img_size[0], self.img_size[1], 3])
        mask.set_shape([self.img_size[0], self.img_size[1], 1])
        return img, mask

    def get_dataset(self, split="train", shuffle=True):
        images_dir = self.root_dir / split / "images"
        masks_dir = self.root_dir / split / "masks"

        # Troviamo i file 
        img_paths = sorted([str(p) for p in images_dir.glob("*")])
        mask_paths = sorted([str(p) for p in masks_dir.glob("*")])

        print(f"--- SPLIT: {split.upper()} ---")
        print("FOUND IMAGES:", len(img_paths))
        print("FOUND MASKS:", len(mask_paths))
        assert len(img_paths) > 0, f"NO IMAGES FOUND in {images_dir}"

        # Creiamo la pipeline nativa di TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)
            
        # Applichiamo il caricamento OpenCV in parallelo
        dataset = dataset.map(self._tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Raggruppiamo in batch
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_split_metadata(self, split="train"):
        """
        Analizza lo split sul disco PRIMA del preprocess e calcola i metadati.
        """
        images_dir = self.root_dir / split / "images"
        masks_dir = self.root_dir / split / "masks"

        img_paths = sorted(list(images_dir.glob("*")))
        mask_paths = sorted(list(masks_dir.glob("*")))

        total_count = len(img_paths)
        if total_count == 0:
            return None

        # Leggiamo la prima immagine per capire la risoluzione ORIGINALE (Pre-preprocess)
        sample_img = cv2.imread(str(img_paths[0]))
        if sample_img is not None:
            orig_h, orig_w, orig_c = sample_img.shape
            orig_resolution = f"{orig_w}x{orig_h}"
        else:
            orig_resolution = "Unknown"
            orig_c = 3

        # Calcoliamo la dimensione totale in Megabyte dello split su disco
        total_bytes = sum(p.stat().st_size for p in img_paths) + sum(p.stat().st_size for p in mask_paths)
        total_mb = round(total_bytes / (1024 * 1024), 2)

        return {
            "count": total_count,
            "size_on_disk_mb": total_mb,
            "pre_preprocessing": {
                "image_resolution": orig_resolution,
                "channels": orig_c,
                "pixel_value_range": "0 - 255 (uint8)"
            },
            "post_preprocessing": {
                "image_resolution": f"{self.img_size[0]}x{self.img_size[1]}",
                "channels": 3,
                "pixel_value_range": "0.0 - 1.0 (float32)",
                "batch_size": self.batch_size,
                "total_batches": int(np.ceil(total_count / self.batch_size))
            }
        }


    def get_sample_preview_b64(self, split="train"):
            """
            Prende un campione CASUALE di uno split, genera la versione originale 
            e la versione preprocessata, convertendole in stringhe Base64.
            """
            images_dir = self.root_dir / split / "images"
            masks_dir = self.root_dir / split / "masks"

            img_paths = sorted(list(images_dir.glob("*")))
            mask_paths = sorted(list(masks_dir.glob("*")))

            if not img_paths or not mask_paths:
                return None

            # Scegliamo un indice casuale tra 0 e il numero totale di immagini disponibili
            random_idx = random.randint(0, len(img_paths) - 1)
            
            # Usiamo lo stesso identico indice per non disallineare immagine e maschera
            chosen_img_path = img_paths[random_idx]
            chosen_mask_path = mask_paths[random_idx]
            # ------------------------

            # 1. Carichiamo i file RAW dal disco usando l'indice casuale
            img_raw = cv2.imread(str(chosen_img_path))
            mask_raw = cv2.imread(str(chosen_mask_path))

            # 2. Applichiamo il preprocess 
            img_proc = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_proc = cv2.resize(img_proc, self.img_size)
            img_proc = (img_proc * 255).astype(np.uint8)
            img_proc = cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR)

            mask_proc = cv2.resize(mask_raw, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask_proc = (mask_proc > 127).astype(np.uint8) * 255

            def encode_b64(img_np, ext=".jpg"):
                _, buffer = cv2.imencode(ext, img_np)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/{ext[1:]};base64,{b64_str}"

            return {
                "metadata": {
                    "original_filename": chosen_img_path.name, 
                    "dataset_index_extracted": random_idx,      
                    "original_shape": f"{img_raw.shape[1]}x{img_raw.shape[0]}",
                    "processed_shape": f"{self.img_size[0]}x{self.img_size[1]}"
                },
                "before_preprocessing": {
                    "image": encode_b64(img_raw, ".jpg"),
                    "mask": encode_b64(mask_raw, ".jpg")
                },
                "after_preprocessing": {
                    "image": encode_b64(img_proc, ".jpg"),
                    "mask": encode_b64(mask_proc, ".png")
                }
            }