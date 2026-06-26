import os
import json
import keras # o import keras a seconda della tua installazione
from flask import Blueprint, jsonify

model_info_bp = Blueprint('model_info', __name__)

# Percorsi dei tuoi file locali
MODEL_PATH = "models/best_unet_model.keras"
METRICS_JSON_PATH = "results/best_metrics.json" # Il tuo JSON salvato

# Carichiamo il modello all'avvio (gestendo le tue funzioni custom per evitare errori)
# Se non hai raggruppato le funzioni in una classe, passa direttamente le funzioni di loss/dice
try:
    # Definiamo dei placeholder generici se Keras richiede la compilazione all'import
    dependencies = {
        "dice_bce_loss": lambda y_true, y_pred: 0.0,
        "dice_coefficient": lambda y_true, y_pred: 0.0
    }
    shared_model = keras.models.load_model(MODEL_PATH, custom_objects=dependencies, compile=False)
    shared_model.build((None, 256, 256, 3))
    print("=== MODELLO .KERAS CARICATO IN MEMORIA CON SUCCESSO ===")
except Exception as e:
    shared_model = None
    print(
        f"WARNING: Impossibile caricare il modello da {MODEL_PATH}. "
        f"L'architettura dettagliata dei layer non sarà disponibile. Errore: {e}"
    )

# =====================================================================
# ENDPOINT 1: ARCHITETTURA DELLA RETE
# =====================================================================
@model_info_bp.route("/model/architecture", methods=["GET"])
def get_model_architecture():
    """
    Restituisce l'elenco sequenziale di tutti i layer del modello
    con i rispettivi tipi e le dimensioni geometriche (shape) di output.
    """
    if shared_model is None:
        return jsonify({
            "status": "error",
            "message": "Architettura non disponibile (file .keras mancante o non inizializzato)."
        }), 503

    try:
        layers_table = []
        for layer in shared_model.layers:
            # Estraiamo la forma geometrica del tensore in uscita in Keras 3
            try:
                out_shape = str(layer.output.shape)
            except Exception:
                try:
                    out_shape = str(layer.output_shape)
                except Exception:
                    out_shape = "Dinamico"

            layers_table.append({
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "output_shape": out_shape,
                "trainable": layer.trainable
            })

        return jsonify({
            "status": "ok",
            "architecture_summary": {
                "model_name": shared_model.name,
                "total_layers": len(shared_model.layers),
                "layers_details": layers_table
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante l'estrazione dell'architettura: {str(e)}"
        }), 500


# =====================================================================
# ENDPOINT 2: PARAMETRI, CONFIGURAZIONE E PERFORMANCE 
# =====================================================================
@model_info_bp.route("/model/stats", methods=["GET"])
def get_model_stats():
    """
    Restituisce in ordine rigoroso: i parametri del modello,
    la configurazione di addestramento e le performance della miglior epoca.
    Llegge direttamente dal report JSON statico.
    """
    if not os.path.exists(METRICS_JSON_PATH):
        return jsonify({
            "status": "error",
            "message": f"File dei metadati '{METRICS_JSON_PATH}' non trovato sul server."
        }), 404

    try:
        with open(METRICS_JSON_PATH, "r") as f:
            json_data = json.load(f)

        # Costruiamo la risposta rispettando l'ordine logico richiesto
        return jsonify({
            "status": "ok",
            
            # 1. PARAMETERS
            "parameters_table": {
                "total_parameters": json_data["model_summary"]["total_parameters"],
                "trainable_parameters": json_data["model_summary"]["trainable_parameters"],
                "non_trainable_parameters": json_data["model_summary"]["non_trainable_parameters"]
            },
            
            # 2. CONFIGURATION
            "configuration": {
                "loss_function": json_data["model_summary"]["loss_function"],
                "monitored_metrics": json_data["model_summary"]["monitored_metrics"]
            },
            
            # 3. PERFORMANCE
            "best_epoch_performance": {
                "train": json_data["best_epoch_metrics"]["train"],
                "validation": json_data["best_epoch_metrics"]["validation"]
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante il caricamento delle statistiche: {str(e)}"
        }),