import tensorflow as tf
import keras
from keras import layers

class CrackDataAugmenter:
    def __init__(self):
        self.augmentation_layers = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2, fill_mode="constant"),
            layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="constant"),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="constant")
        ])

    def _augment_batch(self, img_batch, mask_batch):
        combined = tf.concat([img_batch, mask_batch], axis=-1)
        augmented = self.augmentation_layers(combined)
        return augmented[..., :3], augmented[..., 3:]

    def apply(self, dataset):
        return dataset.map(
            self._augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE
        )