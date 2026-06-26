from flask import Blueprint, jsonify
from it.akron.src.dataset import Crack500LocalDataLoader
from it.akron.state import DATASET_CACHE

dataset_bp = Blueprint("dataset_bp", __name__)

@dataset_bp.route("/dataset/build", methods=["POST"])
def build_dataset():
    loader = Crack500LocalDataLoader(batch_size=16, img_size=(256, 256))

    DATASET_CACHE["train"] = loader.get_dataset("train")
    DATASET_CACHE["val"] = loader.get_dataset("val")
    DATASET_CACHE["test"] = loader.get_dataset("test")

    return jsonify({
        "status": "ok",
        "splits": {
            "train": len(list((loader.root_dir / "train" / "images").glob("*"))),
            "val": len(list((loader.root_dir / "val" / "images").glob("*"))),
            "test": len(list((loader.root_dir / "test" / "images").glob("*")))
        }
    })