import io
import zipfile
import numpy as np
from flask import Blueprint, jsonify, send_file
from it.akron.state import DATASET_CACHE

download_bp = Blueprint('download', __name__)

@download_bp.route("/dataset/download-all", methods=["GET"])
def download_all_datasets():
    """
    Versione ultra-leggera per la RAM. Salva i batch direttamente 
    in un file ZIP man mano che vengono generati, evitando i MemoryError.
    """
    if "train_aug" not in DATASET_CACHE or "val" not in DATASET_CACHE or "test" not in DATASET_CACHE:
        return jsonify({
            "status": "error",
            "message": "Dataset non completamente inizializzato nella cache."
        }), 400

    print("Generazione dello ZIP in streaming per salvare la RAM...")
    
    # Creiamo il file ZIP direttamente in un buffer di memoria
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            
            # Funzione interna per scrivere uno split nel file ZIP a blocchi
            def write_split_to_zip(split_name, tf_dataset):
                batch_idx = 0
                for img_batch, mask_batch in tf_dataset.as_numpy_iterator():
                    # Salviamo i singoli batch come piccoli file .npy indipendenti dentro lo ZIP
                    # Es. dentro lo ZIP avremo: train/images_batch_0.npy
                    img_bytes = io.BytesIO()
                    mask_bytes = io.BytesIO()
                    
                    np.save(img_bytes, img_batch)
                    np.save(mask_bytes, mask_batch)
                    
                    zip_file.writestr(f"{split_name}/images_batch_{batch_idx}.npy", img_bytes.getvalue())
                    zip_file.writestr(f"{split_name}/masks_batch_{batch_idx}.npy", mask_bytes.getvalue())
                    batch_idx += 1
                print(f"Scritto split {split_name} con {batch_idx} batch totali.")

            # Scriviamo i tre split uno alla volta liberando la RAM continuamente
            write_split_to_zip("train", DATASET_CACHE['train_aug'])
            write_split_to_zip("val", DATASET_CACHE['val'])
            write_split_to_zip("test", DATASET_CACHE['test'])

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante la creazione dello ZIP: {str(e)}"
        }), 500

    zip_buffer.seek(0)
    print("ZIP pronto per il download!")
    
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="dataset_crack500_stream.zip"
    )