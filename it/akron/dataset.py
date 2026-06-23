import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Crack500LocalDataLoader:
    """
    Carica il dataset CRACK500 scaricato localmente in una cartella personalizzata.
    Compatibile al 100% con la pipeline Keras + Augmenter.
    """
    def __init__(self, batch_size=16, img_size=(256, 256)):
        # base_dir corrisponde al BASE_DIR / "data" del tuo amico
        self.root_dir = BASE_DIR / "data" 
        self.batch_size = batch_size
        self.img_size = img_size

    def _load_with_opencv(self, img_path, mask_path):
        """
        Usa la logica esatta del tuo amico con OpenCV, avvolta in una funzione 
        compatibile con TensorFlow.
        """
        # Trasformiamo i tensori di stringhe in stringhe Python classiche per cv2
        img_path_str = img_path.numpy().decode('utf-8')
        mask_path_str = mask_path.numpy().decode('utf-8')

        # Logica del tuo amico per l'immagine
        img = cv2.imread(img_path_str)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0

        # Logica del tuo amico per la maschera
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

        # Troviamo i file (esattamente come faceva il tuo amico)
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