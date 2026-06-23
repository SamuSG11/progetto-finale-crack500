import keras.backend as K

class SegmentationMetrics:
    """
    Raccoglie le funzioni di perdita e le metriche di valutazione 
    personalizzate per la segmentazione binaria delle crepe.
    """
    @staticmethod
    def dice_bce_loss(y_true, y_pred):
        """Loss ibrida che combina Binary Cross Entropy e Dice Loss."""
        smooth = 1e-6
        # BCE pixel-per-pixel
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Dice Loss geometrica
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        return bce + (1.0 - dice_score)

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        """Metrica di monitoraggio del coefficiente Dice (soglia a 0.5)."""
        smooth = 1e-6
        y_pred_bin = K.cast(K.greater(y_pred, 0.5), dtype="float32")
        
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred_bin)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)