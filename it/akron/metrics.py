import tensorflow as tf
import keras

class SegmentationMetrics:
    @staticmethod
    def dice_bce_loss(y_true, y_pred):
        smooth = 1e-6
        # Calcolo della Binary Crossentropy standard di Keras
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Appiattiamo i tensori usando tf.reshape
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Calcolo del Dice Coefficient con le ops di Keras 3
        intersection = keras.ops.sum(y_true_f * y_pred_f)
        dice_score = (2. * intersection + smooth) / (keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) + smooth)
        
        return bce + (1.0 - dice_score)

    @staticmethod
    def dice_coefficient(y_true, y_pred):
        smooth = 1e-6
        
        y_pred_bin = tf.cast(tf.greater(y_pred, 0.5), dtype="float32")
        
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred_bin, [-1])
        
        intersection = keras.ops.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (keras.ops.sum(y_true_f) + keras.ops.sum(y_pred_f) + smooth)