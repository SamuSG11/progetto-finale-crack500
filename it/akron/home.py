from flask import Blueprint

home_bp = Blueprint("home", __name__)


@home_bp.route("/", methods=["GET"])
def intro():
    return """
        <h1>--- API CRACK 500 - IMAGE SEGMENTATION ---</h1>

        <p>
            REST API per la gestione del dataset CRACK500 e pipeline di data augmentation
            per task di image segmentation (crack detection).
        </p>
        <p>
            Home alternativa visibile su /home
        </p>
        
        <hr>
        
        <h2>CARICAMENTO DEL DATASET:</h2>

        <h3>BUILD DATASET</h3>
        <ul>
            <li><b>POST /dataset/build</b></li>
            <p>Carica il dataset CRACK500 e crea i dataset TensorFlow per train, validation e test.</p>
        </ul>
        
        <hr>

       <h2>DATA AUGMENTATION</h2>

        <h3>AUGMENT DEL DATASET</h3>
        <ul>
            <li><b>POST /dataset/augment</b></li>
            <p>Applica data augmentation sincronizzata (immagini + maschere) al training set.</p>
            <p>Include trasformazioni geometriche: flip, rotation, zoom, translation.</p>
        </ul>

        <h3>INFO AUGMENTATION</h3>
        <ul>
            <li><b>GET /dataset/preview_aug</b></li>
            <p>Mostra informazioni su un batch augmentato (shape immagini e maschere).</p>
            <p>Utile per debugging della pipeline di augmentation.</p>
        </ul>

        <hr>
        
        <h2>RETE NEURALE - U-NET (DA IMPLEMENTARE):</h2>

        <p>...</p>
        <ul>
            <li>...</li>
            <p>p...</p>
        </ul>

        """

@home_bp.route("/home", methods=["GET"])
def intro_detailed():
    return """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #0f172a;
                color: #e2e8f0;
                margin: 40px;
            }

            h1 {
                color: #38bdf8;
            }

            h2 {
                border-left: 4px solid #38bdf8;
                padding-left: 10px;
                margin-top: 30px;
            }

            h3 {
                color: #a5b4fc;
            }

            .endpoint {
                background-color: #1e293b;
                padding: 10px;
                border-radius: 8px;
                font-family: monospace;
                color: #fbbf24;
                display: inline-block;
                margin: 5px 0;
            }

            .box {
                background-color: #111827;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
            }

            .note {
                color: #94a3b8;
                font-size: 14px;
            }

            pre {
                background-color: #1e293b;
                padding: 10px;
                border-radius: 8px;
                overflow-x: auto;
            }
        </style>
    </head>

    <body>

        <h1>🧠 CRACK500 API - IMAGE SEGMENTATION</h1>

        <div class="box">
            <p>REST API per dataset CRACK500 e pipeline di data augmentation per segmentation.</p>
        </div>

        <h2>📦 DATASET PIPELINE</h2>

        <div class="box">
            <h3>1. Build Dataset</h3>
            <div class="endpoint">POST /dataset/build</div>
            <p>Carica dataset e crea train/val/test TensorFlow pipeline.</p>
        </div>

        <div class="box">
            <h3>2. Dataset Info</h3>
            <div class="endpoint">GET /dataset/info</div>
            <p>Statistiche dataset e stato augmentation.</p>
        </div>

        <h2>⚙️ DATA AUGMENTATION</h2>

        <div class="box">
            <h3>3. Apply Augmentation</h3>
            <div class="endpoint">POST /dataset/augment</div>
            <p>Applica trasformazioni geometriche su train set.</p>
        </div>

        <div class="box">
            <h3>4. Preview Augmentation</h3>
            <div class="endpoint">GET /dataset/preview_aug</div>
            <p>Mostra shape batch augmentato (debug pipeline).</p>
        </div>

        <h2>📊 PIPELINE FLOW</h2>

        <pre>
/dataset/build
      ↓
DATASET_CACHE
      ↓
/dataset/augment
      ↓
train_aug
      ↓
/dataset/preview_aug
        </pre>

        <p class="note">
            ⚠ Dataset gestito con tf.data.Dataset → non serializzabile in JSON.
        </p>

    </body>
    </html>
    """