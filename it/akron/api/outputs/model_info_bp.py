import os
import json
import keras
from flask import Blueprint, jsonify, request
from it.akron.src.performance_plot import PerformanceVisualizer


model_info_bp = Blueprint('model_info', __name__)

# =====================================================================
# CONFIGURAZIONE REGISTRO MODELLI DIZIONARIO DINAMICO
# =====================================================================
MODELS_CONFIG = {
    "efficientnet_unet": {
        "model_path": "models/best_efficientnet_unet.keras",
        "json_stats_path": "results/best_unet_efficient_metrics.json",
        "json_test_path": "results/unet_efficient_test_metrics.json" # Nuovo file di test
    },
    "unet_base": {
        "model_path": "models/best_unet_model.keras",
        "json_stats_path": "results/best_unet_metrics.json",
        "json_test_path": "results/unet_test_metrics.json" # File di test del modello base
    }
}

# Dizionario per memorizzare i modelli caricati in RAM all'avvio
loaded_models = {}

# Definiamo dei placeholder generici se Keras richiede la compilazione all'import
dependencies = {
    "dice_bce_loss": lambda y_true, y_pred: 0.0,
    "dice_coefficient": lambda y_true, y_pred: 0.0
}

# Caricamento automatico di tutti i modelli configurati all'avvio del server
for model_key, config in MODELS_CONFIG.items():
    try:
        if os.path.exists(config["model_path"]):
            model = keras.models.load_model(config["model_path"], custom_objects=dependencies, compile=False)
            model.build((None, 256, 256, 3))
            loaded_models[model_key] = model
            print(f"=== MODELLO '{model_key}' CARICATO IN MEMORIA CON SUCCESSO ===")
        else:
            print(f"WARNING: File non trovato per {model_key} al percorso {config['model_path']}")
    except Exception as e:
        print(f"ERROR: Impossibile caricare il modello {model_key}. Errore: {e}")


# =====================================================================
# ENDPOINT 1: ARCHITETTURA DELLA RETE (DINAMICO)
# =====================================================================
@model_info_bp.route("/model/architecture", methods=["GET"])
def get_model_architecture():
    """
    Restituisce l'elenco sequenziale di tutti i layer del modello richiesto
    con i rispettivi tipi e le dimensioni geometriche (shape) di output.
    Query param accettati: ?name=efficientnet_unet (Default) o ?name=unet_base
    """
    model_name = request.args.get('name', 'efficientnet_unet').lower()

    if model_name not in loaded_models:
        return jsonify({
            "status": "error",
            "message": f"Architettura non disponibile o modello '{model_name}' non caricato. Modelli disponibili: {list(loaded_models.keys())}"
        }), 400

    target_model = loaded_models[model_name]

    try:
        layers_table = []
        for layer in target_model.layers:
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
            "model_selected": model_name,
            "architecture_summary": {
                "model_name": target_model.name,
                "total_layers": len(target_model.layers),
                "layers_details": layers_table
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante l'estrazione dell'architettura di {model_name}: {str(e)}"
        }), 500


# =====================================================================
# ENDPOINT 2: PARAMETRI, CONFIGURAZIONE E PERFORMANCE VAL/TRAIN
# =====================================================================
@model_info_bp.route("/model/stats", methods=["GET"])
def get_model_stats():
    """
    Restituisce i parametri del modello richiesto, la configurazione di 
    addestramento e le performance della miglior epoca (Train/Val).
    Query param accettati: ?name=efficientnet_unet (Default) o ?name=unet_base
    """
    model_name = request.args.get('name', 'efficientnet_unet').lower()

    if model_name not in MODELS_CONFIG:
        return jsonify({"status": "error", "message": f"Modello '{model_name}' non configurato."}), 400

    json_path = MODELS_CONFIG[model_name]["json_stats_path"]

    if not os.path.exists(json_path):
        return jsonify({"status": "error", "message": f"File dei metadati '{json_path}' non trovato."}), 404

    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)

        return jsonify({
            "status": "ok",
            "model_selected": model_name,
            "parameters_table": {
                "total_parameters": json_data["model_summary"]["total_parameters"],
                "trainable_parameters": json_data["model_summary"]["trainable_parameters"],
                "non_trainable_parameters": json_data["model_summary"]["non_trainable_parameters"]
            },
            "configuration": {
                "loss_function": json_data["model_summary"]["loss_function"],
                "monitored_metrics": json_data["model_summary"]["monitored_metrics"]
            },
            "best_epoch_performance": {
                "train": json_data["best_epoch_metrics"]["train"],
                "validation": json_data["best_epoch_metrics"]["validation"]
            }
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Errore caricamento statistiche: {str(e)}"}), 500


# =====================================================================
# ENDPOINT 3: PERFORMANCE SUL TEST SET (NUOVO)
# =====================================================================
@model_info_bp.route("/model/test-performance", methods=["GET"])
def get_model_test_performance():
    """
    Restituisce le metriche reali ottenute dal modello sul Test Set (dati mai visti).
    Query param accettati: ?name=efficientnet_unet (Default) o ?name=unet_base
    """
    model_name = request.args.get('name', 'efficientnet_unet').lower()

    if model_name not in MODELS_CONFIG:
        return jsonify({"status": "error", "message": f"Modello '{model_name}' non configurato."}), 400

    # Peschiamo il percorso specifico del file di test registrato nel dizionario
    json_test_path = MODELS_CONFIG[model_name]["json_test_path"]

    if not os.path.exists(json_test_path):
        return jsonify({
            "status": "error", 
            "message": f"File delle metriche di test '{json_test_path}' non trovato sul server per il modello '{model_name}'."
        }), 404

    try:
        with open(json_test_path, "r") as f:
            test_data = json.load(f)

        return jsonify({
            "status": "ok",
            "model_selected": model_name,
            # Restituisce direttamente il blocco delle performance sul test set generato da Colab
            "test_performance": test_data.get("test_performance", test_data)
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante la lettura del file di test per {model_name}: {str(e)}"
        }), 500



from flask import send_file
# Se sposti la classe in un altro file (es. utils/visualizer.py), scommenta la riga sotto:
# from utils.visualizer import PerformanceVisualizer

# Istanziamo la classe di servizio passandogli il dizionario MODELS_CONFIG che hai già definito in cima al file
visualizer_service = PerformanceVisualizer(MODELS_CONFIG)

@model_info_bp.route("/model/compare-performance-chart", methods=["GET"])
def get_comparison_chart():
    """
    Endpoint che genera dinamicamente il grafico a barre raggruppate (Grouped Bar Chart)
    confrontando Train, Validation e Test Dice Coefficient di tutti i modelli configurati.
    """
    try:
        # Generiamo il grafico tramite la classe
        chart_buffer = visualizer_service.generate_comparison_chart()
        
        # Restituiamo direttamente il file binario dell'immagine PNG al client
        return send_file(chart_buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante la generazione del grafico comparativo: {str(e)}"
        }), 500