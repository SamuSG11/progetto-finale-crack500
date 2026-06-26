from flask import Blueprint, jsonify
from it.akron.src.augmentation import CrackDataAugmenter
from it.akron.state import DATASET_CACHE

augment_bp = Blueprint("augment_bp", __name__)

@augment_bp.route("/dataset/augment", methods=["POST"])
def augment_dataset():

    if "train" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset non inizializzato"
        }), 400

    augmenter = CrackDataAugmenter()

    DATASET_CACHE["train_aug"] = augmenter.apply(
        DATASET_CACHE["train"]
    )

    return jsonify({
        "status": "ok"
    })

@augment_bp.route("/dataset/preview_aug", methods=["GET"])
def preview_augmented():

    if "train_aug" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Augmentation non ancora applicata"
        }), 400

    batch = next(iter(DATASET_CACHE["train_aug"]))

    imgs, masks = batch

    return jsonify({
        "status": "ok",
        "image_shape": list(imgs.shape),
        "mask_shape": list(masks.shape),
        "note": "batch TensorFlow non serializzato completamente"
    })