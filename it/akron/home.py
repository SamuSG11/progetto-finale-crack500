from flask import Blueprint

home_bp = Blueprint("home", __name__)


@home_bp.route("/", methods=["GET"])
def intro():
    return """
        <h1>--- API CRACK 500 - IMAGE SEGMENTATION ---</h1>

        <p>
            REST API per l'analisi del dataset di immagini Crack500
        </p>

        <h2>CARICAMENTO DEL DATASET:</h2>

        <h3>Dataset load</h3>
        <ul>
            <li>POST /load
            <p> ...</p>
            <p> ___</p>
        </ul>

        <h2>PREPROCESSING DEL DATASET:</h2>

        <h3>Preprocessing del dataset prima di creare la u-net</h3>
        <ul>
            <li>...</li>
            <li>...</li>
        </ul>

        <h2>RETE NEURALE - U-NET:</h2>

        <p>...</p>
        <ul>
            <li>...</li>
            <p>p...</p>
        </ul>

        """