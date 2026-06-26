from flask import Blueprint, jsonify, request
from it.akron.dataset import Crack500LocalDataLoader

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