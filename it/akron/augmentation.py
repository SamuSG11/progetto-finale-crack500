import tensorflow as tf
import keras
from keras import layers

class CrackDataAugmenter:
    """Classe indipendente dedicata esclusivamente alla Data Augmentation geometrica."""
    def __init__(self):
        # Configurazione dei livelli di deformazione casuale su GPU
        self.augmentation_layers = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2, fill_mode="constant"),
            layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="constant"),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant")
        ])

    def _augment_batch(self, img_batch, mask_batch):
        """Prende un intero pacchetto (batch) e lo deforma in sincrono."""
        # Uniamo i canali lungo l'ultimo asse per incollare l'immagine alla sua maschera
        combined = tf.concat([img_batch, mask_batch], axis=-1)
        # Deformiamo l'intero blocco
        combined_augmented = self.augmentation_layers(combined)
        # Separiamo nuovamente l'immagine (canali 0,1,2) dalla maschera (canale 3)
        return combined_augmented[..., :3], combined_augmented[..., 3:]

    def apply(self, dataset):
        """Prende un tf.data.Dataset in ingresso e gli inietta la data augmentation."""
        return dataset.map(self._augment_batch, num_parallel_calls=tf.data.AUTOTUNE)