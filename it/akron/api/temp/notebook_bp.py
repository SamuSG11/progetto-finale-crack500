from flask import Blueprint, jsonify


notebook_bp = Blueprint('notebook', __name__)

@notebook_bp.route("/notebook/train", methods=["GET"])
def notebook():
    """
    Endpoint che ritorna il link del notebook in colab per addestrare il modello.
    """
    return jsonify({
        "link-unet: https://colab.research.google.com/drive/1-oHrzo_ILGN4Vn6ZO_NK2vLI-ObPcE7I#scrollTo=SjYICXnBnlGY"
        "link-efficientnet: https://colab.research.google.com/drive/13ihmNUVQmJwvbS6XNG-4RuaEIGxh9VcD#scrollTo=z-xvSYs2UkEv"
    }), 200