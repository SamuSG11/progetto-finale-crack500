from flask import Blueprint

home_bp = Blueprint("home", __name__)

@home_bp.route("/", methods=["GET"])
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

        <h1>CRACK500 API - IMAGE SEGMENTATION</h1>

        <div class="box">
            <p>REST API per dataset CRACK500 e pipeline di data augmentation per segmentation.</p>
        </div>

        <h2>📦 DATASET PIPELINE</h2>

        <div class="box">
            <h3>Build Dataset</h3>
            <div class="endpoint">POST /dataset/build</div>
            <p>Carica dataset e crea train/val/test TensorFlow pipeline.</p>
        </div>

        <div class="box">
            <h3>Dataset Info</h3>
            <div class="endpoint">GET /dataset/info</div>
            <p>Statistiche dataset e stato augmentation.</p>
        </div>

        <h2>⚙️ DATA AUGMENTATION</h2>

        <div class="box">
            <h3>Apply Augmentation</h3>
            <div class="endpoint">POST /dataset/augment</div>
            <p>Applica trasformazioni geometriche su train set.</p>
        </div>

        <div class="box">
            <h3>Preview Augmentation</h3>
            <div class="endpoint">GET /dataset/preview_aug</div>
            <p>Mostra shape batch augmentato (debug pipeline).</p>
        </div>
        
        <h2>📓 TRAINING NOTEBOOK</h2>

        <div class="box">
            <h3>Open Colab Notebook</h3>
            <div class="endpoint">GET /notebook/train</div>
            <p>
                Restituisce il link al notebook Google Colab utilizzato per il training
                del modello sul dataset CRACK500.
            </p>
        </div>
        
        <h2>🚀 DATASET OPTIMIZATION</h2>

        <div class="box">
            <h3>Enable Prefetch</h3>
            <div class="endpoint">POST /dataset/prefetch</div>
            <p>
                Ottimizza la pipeline TensorFlow applicando il prefetch ai dataset,
                migliorando le prestazioni durante il training riducendo i tempi di attesa
                tra il caricamento dei batch e l'esecuzione del modello.
            </p>
        </div>

        <h2>✅ DATASET VALIDATION</h2>

        <div class="box">
            <h3>Verify Dataset</h3>
            <div class="endpoint">POST /dataset/verify</div>
            <p>
                Verifica il corretto caricamento del dataset e la coerenza dei dati,
                controllando immagini, maschere, dimensioni e corrispondenza tra i campioni
                della pipeline TensorFlow.
            </p>
        </div>
        
        <h2>🧠 MODEL ANALYSIS</h2>

        <div class="box">
            <h3>Model Architecture</h3>
            <div class="endpoint">GET /model/architecture</div>
            <p>
                Restituisce l'architettura del modello richiesto, elencando tutti i layer,
                le dimensioni degli output e il numero di parametri. Il parametro
                <code>name</code> accetta <b>efficientnet_unet</b> oppure <b>unet_base</b>.
            </p>
        </div>

        <div class="box">
            <h3>Model Statistics</h3>
            <div class="endpoint">GET /model/stats</div>
            <p>
                Restituisce le statistiche del modello selezionato, inclusi numero di
                parametri, configurazione di training, iperparametri e le performance
                ottenute alla migliore epoca sui set di training e validation.
            </p>
        </div>
        
        <div class="box">
            <h3>Best Epoch Metrics</h3>
            <div class="endpoint">GET /model/best-metrics</div>
            <p>
                Restituisce i parametri del modello e calcola dinamicamente le metriche
                relative alla migliore epoca di training utilizzando i dati salvati in
                cache, senza rieseguire l'addestramento del modello.
            </p>
        </div>

        <div class="box">
            <h3>Test Set Performance</h3>
            <div class="endpoint">GET /model/test-performance</div>
            <p>
                Restituisce le metriche di valutazione del modello sul test set,
                come Loss, Accuracy, Precision, Recall, F1-Score, IoU e Dice Score.
            </p>
        </div>

        <div class="box">
            <h3>Compare Model Performance</h3>
            <div class="endpoint">GET /model/compare-performance-chart</div>
            <p>
                Restituisce un grafico a barre raggruppate che confronta le principali
                metriche di valutazione dei modelli disponibili sul test set,
                facilitandone l'analisi comparativa.
            </p>
        </div>

    </body>
    </html>
    """