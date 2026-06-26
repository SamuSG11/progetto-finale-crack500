import keras
from keras import layers

class UNetStandard:
    """
    Classe per la costruzione di una rete U-Net classica per la segmentazione di immagini.
    """
    def __init__(self, input_shape=(256, 256, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _double_conv_block(self, x, n_filters):
        """
        Blocco base ripetuto: due convoluzioni consecutive da 3x3,
        ciascuna seguita da Batch Normalization e attivazione ReLU.
        """
        # Prima Convoluzione
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Seconda Convoluzione
        x = layers.Conv2D(n_filters, kernel_size=3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def build_model(self):
        """Costruisce e restituisce il modello Keras completo."""
        inputs = layers.Input(shape=self.input_shape)

        # ==========================================
        # 1. ENCODER (Fase di contrazione)
        # ==========================================
        # Blocco 1: In 256x256x3 -> Out 256x256x64 (Dimezza a 128x128)
        c1 = self._double_conv_block(inputs, 64)
        p1 = layers.MaxPooling2D(pool_size=2)(c1)

        # Blocco 2: In 128x128x64 -> Out 128x128x128 (Dimezza a 64x64)
        c2 = self._double_conv_block(p1, 128)
        p2 = layers.MaxPooling2D(pool_size=2)(c2)

        # Blocco 3: In 64x64x128 -> Out 64x64x256 (Dimezza a 32x32)
        c3 = self._double_conv_block(p2, 256)
        p3 = layers.MaxPooling2D(pool_size=2)(c3)

        # Blocco 4: In 32x32x256 -> Out 32x32x512 (Dimezza a 16x16)
        c4 = self._double_conv_block(p3, 512)
        p4 = layers.MaxPooling2D(pool_size=2)(c4)

        # ==========================================
        # 2. BOTTLENECK (Il punto più profondo)
        # ==========================================
        # In 16x16x512 -> Out 16x16x1024
        bottleneck = self._double_conv_block(p4, 1024)

        # ==========================================
        # 3. DECODER (Fase di espansione)
        # ==========================================
        # Blocco 1: Ingrandisce il bottleneck (a 32x32) e lo concatena con c4
        u1 = layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding="same")(bottleneck)
        u1 = layers.Concatenate()([u1, c4]) # Skip Connection
        c5 = self._double_conv_block(u1, 512)

        # Blocco 2: Ingrandisce (a 64x64) e lo concatena con c3
        u2 = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding="same")(c5)
        u2 = layers.Concatenate()([u2, c3]) # Skip Connection
        c6 = self._double_conv_block(u2, 256)

        # Blocco 3: Ingrandisce (a 128x128) e lo concatena con c2
        u3 = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding="same")(c6)
        u3 = layers.Concatenate()([u3, c2]) # Skip Connection
        c7 = self._double_conv_block(u3, 128)

        # Blocco 4: Ingrandisce (a 256x256) e lo concatena con c1
        u4 = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding="same")(c7)
        u4 = layers.Concatenate()([u4, c1]) # Skip Connection
        c8 = self._double_conv_block(u4, 64)

        # ==========================================
        # 4. SEGMENTATION HEAD (Output finale)
        # ==========================================
        # Usiamo 1 filtro finale con attivazione Sigmoid: 
        # restituisce una probabilità da 0.0 a 1.0 per ogni singolo pixel.
        outputs = layers.Conv2D(self.num_classes, kernel_size=1, activation="sigmoid")(c8)

        # Istanziamo il modello Keras definitivo
        model = keras.Model(inputs=inputs, outputs=outputs, name="U-Net_Standard_Crack500")
        return model