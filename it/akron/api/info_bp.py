from flask import Blueprint, jsonify, request
# Assicurati di importare correttamente la classe dal tuo file
from it.akron.dataset import Crack500LocalDataLoader 

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route("/dataset/info", methods=["GET"])
def get_dataset_info():
    """
    Endpoint che restituisce la panoramica completa del dataset CRACK500
    mostrando lo stato dei dati prima e dopo il ciclo di preprocessing.
    """
    try:
        # Inizializziamo il loader con i parametri standard del progetto
        loader = Crack500LocalDataLoader(batch_size=16, img_size=(256, 256))
        
        splits = ["train", "val", "test"]
        dataset_report = {}
        global_total_images = 0
        global_total_mb = 0.0

        # Raccogliamo i metadati per ogni cartella
        for split in splits:
            metadata = loader.get_split_metadata(split)
            if metadata:
                dataset_report[split] = metadata
                global_total_images += metadata["count"]
                global_total_mb += metadata["size_on_disk_mb"]
            else:
                dataset_report[split] = {"status": "empty or directory missing"}

        # Costruiamo il JSON di risposta finale
        return jsonify({
            "status": "ok",
            "dataset_name": "CRACK500",
            "summary": {
                "total_samples": global_total_images,
                "total_dataset_size_mb": round(global_total_mb, 2),
                "data_loader_class": "Crack500LocalDataLoader"
            },
            "splits": dataset_report
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Impossibile scansionare i metadati del dataset: {str(e)}"
        }), 500



@dataset_bp.route("/dataset/preview-samples", methods=["GET"])
def get_dataset_preview():
    """
    Endpoint che restituisce le stringhe Base64 delle immagini e delle maschere
    prima e dopo il trattamento di preprocessing di OpenCV.
    Inviare query param '?split=val' per cambiare cartella (default: train).
    """
    try:
        # Recuperiamo lo split richiesto dall'URL (es: /preview-samples?split=val)
        split_requested = request.args.get('split', 'train').lower()
        
        if split_requested not in ['train', 'val', 'test']:
            return jsonify({"status": "error", "message": "Split non valido. Usa train, val o test."}), 400

        loader = Crack500LocalDataLoader(batch_size=16, img_size=(256, 256))
        preview_data = loader.get_sample_preview_b64(split=split_requested)

        if not preview_data:
            return jsonify({
                "status": "error", 
                "message": f"Nessun file trovato per lo split '{split_requested}'."
            }), 404

        return jsonify({
            "status": "ok",
            "split": split_requested,
            "data": preview_data
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante la generazione della preview: {str(e)}"
        }), 500