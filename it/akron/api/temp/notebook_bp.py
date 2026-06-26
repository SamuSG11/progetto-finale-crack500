from flask import Blueprint, jsonify


notebook_bp = Blueprint('notebook', __name__)

@notebook_bp.route("/notebook/train", methods=["GET"])
def notebook():
    """
    Endpoint che ritorna il link del notebook in colab per addestrare il modello.
    """
    return jsonify({
        "status": "ok",
        "message": "Notebook API is running."
        "link"
    }), 200