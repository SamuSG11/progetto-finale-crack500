import numpy as np
from flask import Blueprint, jsonify
from it.akron.state import DATASET_CACHE


verify_bp = Blueprint('verify', __name__)
 

@verify_bp.route("/dataset/verify", methods=["POST"])
def verify_dataset():
    """
    Endpoint per eseguire il test di verifica sul caricamento e sulla coerenza
    dei dati (dimensioni, normalizzazione e assenza di NaN).
    Prende in esame il dataset aumentato 'train_aug' presente in cache.
    """
    
    # 1. Controllo di sicurezza: il dataset di train aumentato è in cache?
    if "train_aug" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset 'train_aug' non trovato in cache. Inizializzalo prima di eseguire il test."
        }), 400

    train_ds = DATASET_CACHE["train_aug"]

    try:
        # Estraiamo un solo batch dal train_ds usando .take(1)
        # Usiamo un flag per essere sicuri che il ciclo for estragga effettivamente qualcosa
        batch_extracted = False
        
        for test_images, test_masks in train_ds.take(1):
            batch_extracted = True
            
            # Convertiamo i tensori in array numpy per analizzarli facilmente
            img_np = test_images.numpy()
            mask_np = test_masks.numpy()
            
            # -----------------------------------------------------------------
            # [TEST 1] Verifica Dimensioni (Atteso: batch_size=16, 256, 256, canali)
            # -----------------------------------------------------------------
            expected_img_shape = (16, 256, 256, 3)
            expected_mask_shape = (16, 256, 256, 1)
            
            if img_np.shape != expected_img_shape:
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 1: Dimensioni delle immagini",
                    "message": f"Errore nelle dimensioni delle immagini! Atteso {expected_img_shape}, rilevato {img_np.shape}"
                }), 400
                
            if mask_np.shape != expected_mask_shape:
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 1: Dimensioni delle maschere",
                    "message": f"Errore nelle dimensioni delle maschere! Atteso {expected_mask_shape}, rilevato {mask_np.shape}"
                }), 400

            # -----------------------------------------------------------------
            # [TEST 2] Verifica Normalizzazione (Atteso: valori tra 0.0 e 1.0)
            # -----------------------------------------------------------------
            img_min, img_max = float(img_np.min()), float(img_np.max())
            mask_min, mask_max = float(mask_np.min()), float(mask_np.max())
            
            if not (0.0 <= img_min and img_max <= 1.0):
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 2: Normalizzazione Immagini",
                    "message": f"Le immagini non sono normalizzate tra [0, 1]! Rilevato Min={img_min}, Max={img_max}"
                }), 400
                
            if not (0.0 <= mask_min and mask_max <= 1.0):
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 2: Normalizzazione Maschere",
                    "message": f"Le maschere non sono normalizzate tra [0, 1]! Rilevato Min={mask_min}, Max={mask_max}"
                }), 400

            # -----------------------------------------------------------------
            # [TEST 3] Verifica Coerenza Matematica (Atteso: No NaN o infiniti)
            # -----------------------------------------------------------------
            if np.isnan(img_np).any():
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 3: Coerenza Matematica Immagini",
                    "message": "Rilevati valori NaN nelle immagini dopo l'augmentation!"
                }), 400
                
            if np.isnan(mask_np).any():
                return jsonify({
                    "status": "error",
                    "test_failed": "TEST 3: Coerenza Matematica Maschere",
                    "message": "Rilevati valori NaN nelle maschere dopo l'augmentation!"
                }), 400

        # Se il ciclo for non è mai partito significa che il dataset è vuoto
        if not batch_extracted:
            return jsonify({
                "status": "error",
                "message": "Il dataset 'train_aug' è vuoto, impossibile estrarre un batch per il test."
            }), 400

        # Se tutti i controlli if passano, restituiamo il successo con i dati dei test
        return jsonify({
            "status": "ok",
            "message": "TUTTI I TEST SONO STATI SUPERATI CON SUCCESSO! Pipeline dati verificata.",
            "report": {
                "test_1_dimensions": {
                    "status": "PASSED",
                    "images_shape": list(img_np.shape),
                    "masks_shape": list(mask_np.shape)
                },
                "test_2_normalization": {
                    "status": "PASSED",
                    "images_range": {"min": img_min, "max": img_max},
                    "masks_range": {"min": mask_min, "max": mask_max}
                },
                "test_3_mathematical_coherence": {
                    "status": "PASSED",
                    "has_nan_or_inf": False
                }
            }
        }), 200

    except Exception as e:
        # Cattura eventuali errori imprevisti (es. problemi di memoria o di conversione)
        return jsonify({
            "status": "error",
            "message": f"Errore imprevisto durante l'esecuzione dei test: {str(e)}"
        }), 500