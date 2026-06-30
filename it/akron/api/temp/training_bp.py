import keras
from flask import Blueprint, jsonify
from it.akron.state import DATASET_CACHE
from it.akron.src.metrics import SegmentationMetrics


training_bp = Blueprint('training', __name__)

MODEL_PATH = "models/best_unet_model.keras"

@training_bp.route("/model/best-metrics", methods=["GET"])
def get_model_best_metrics():
    """
    Endpoint per estrarre i parametri del modello e calcolare al volo
    le metriche della 'best epoch' usando i dati salvati in cache.
    """

    print("--- DEBUG CAPACITÀ CACHE ---")
    print("Chiavi attualmente presenti in DATASET_CACHE:", list(DATASET_CACHE.keys()))
    print("ID della memoria del dizionario:", id(DATASET_CACHE))
    print("----------------------------")

    if "train_opt" not in DATASET_CACHE or "val_opt" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": f"Dataset non trovati. Chiavi in cache: {list(DATASET_CACHE.keys())}"
        }), 400

    # 1. Controllo di sicurezza sui dati in cache
    if "train_opt" not in DATASET_CACHE or "val_opt" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset (train_opt o val_opt) non trovati in cache. Caricali prima di valutare."
        }), 400

    try:
        custom_mapping = {
            "dice_bce_loss": SegmentationMetrics.dice_bce_loss,
            "dice_coefficient": SegmentationMetrics.dice_coefficient
        }
        # 2. Caricamento del modello (includi custom_objects se usi metriche personalizzate)
        # Sostituisci con le tue funzioni reali (es. Dice Coefficient) se necessario
        model = keras.models.load_model(MODEL_PATH, compile=True, custom_objects=custom_mapping)
        
        # 3. Estrazione dei Parametri Strutturali
        # Contiamo i parametri totali, addestrabili e non addestrabili
        total_params = model.count_params()
        trainable_params = sum([keras.ops.prod(w.shape) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        # 4. Calcolo delle metriche della Best Epoch (Valutazione al volo)
        print("Calcolo metriche sul Train Set...")

        # Valutiamo solo su un sottoinsieme di batch (es. 12 batch da 16 = ~200 immagini)
        NUM_BATCH_TEST = 1

        print("Calcolo rapido metriche sul Train Set (Campionato)...")
        train_results = model.evaluate(DATASET_CACHE["train_opt"].take(NUM_BATCH_TEST), verbose=0)
        train_metrics = dict(zip(model.metrics_names, train_results))

        print("Calcolo rapido metriche sul Validation Set (Campionato)...")
        val_results = model.evaluate(DATASET_CACHE["val_opt"].take(NUM_BATCH_TEST), verbose=0)
        val_metrics = dict(zip(model.metrics_names, val_results))

        # 5. Generazione del Report Finale
        return jsonify({
            "status": "ok",
            "model_summary": {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(non_trainable_params),
                "loss_function": str(model.loss),
                "monitored_metrics": model.metrics_names
            },
            "best_epoch_metrics": {
                "train": {k: float(v) for k, v in train_metrics.items()},
                "validation": {k: float(v) for k, v in val_metrics.items()}
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante l'estrazione delle info dal modello: {str(e)}"
        }), 500