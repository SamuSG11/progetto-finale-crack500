from flask import Blueprint, jsonify, request
import tensorflow as tf
from it.akron.state import DATASET_CACHE

optimize_bp = Blueprint('optimize', __name__)

@optimize_bp.route("/dataset/prefetch", methods=["POST"])
def prefetch_dataset():
    """
    Endpoint per ottimizzare un dataset in cache applicando il prefetch di TensorFlow.
    """
    if "test" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset di test non inizializzato"
        }), 400

    if "train_aug" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "train_aug non inizializzato"
        }), 400

    if "val" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset di validation non inizializzato"
        }), 400

    try:
        DATASET_CACHE['train_opt'] = DATASET_CACHE['train_aug'].prefetch(buffer_size=tf.data.AUTOTUNE)
        DATASET_CACHE['val_opt'] = DATASET_CACHE['val'].prefetch(buffer_size=tf.data.AUTOTUNE)
        DATASET_CACHE['test_opt'] = DATASET_CACHE['test'].prefetch(buffer_size=tf.data.AUTOTUNE)
    except AttributeError:
        # Se gli oggetti in cache non fossero tf.data.Dataset ma liste/array, il prefetch fallirebbe.
        return jsonify({
            "status": "error",
            "message": "Errore: Gli oggetti in cache non sono istanze di tf.data.Dataset valide."
        }), 500
    return jsonify({
        "status": "ok",
        "message": "Prefetch applicato con successo a train_aug, val e test."
    }), 200