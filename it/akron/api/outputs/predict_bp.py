import io
import cv2
import numpy as np
import keras # o import keras a seconda della tua installazione
from flask import Blueprint, request, jsonify, send_file

predict_bp = Blueprint('predict', __name__)

# Assumiamo che il modello sia caricato globalmente all'avvio
MODEL_PATH = "models/best_unet_model.keras"
dependencies = {
    "dice_bce_loss": lambda y_true, y_pred: 0.0,
    "dice_coefficient": lambda y_true, y_pred: 0.0
}
model = keras.models.load_model(MODEL_PATH, custom_objects=dependencies, compile=False)

@predict_bp.route("/predict/comparison-overlay", methods=["POST"])
def predict_comparison_overlay():
    """
    Endpoint che riceve un'immagine di input, predice la maschera delle crepe
    e restituisce l'immagine originale con la maschera rossa in sovrimpressione.
    """
    # 1. Controllo di sicurezza: il file è presente nella richiesta?
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Nessun file immagine inviato nella chiave 'image'."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "File non valido o vuoto."}), 400

    try:
        # 2. Leggiamo il file inviato e convertiamolo in un'immagine OpenCV
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Immagine BGR originale
        
        if img_original is None:
            return jsonify({"status": "error", "message": "Il file inviato non è un'immagine valida."}), 400

        # Salviamo le dimensioni originali per ripristinarle alla fine
        orig_h, orig_w = img_original.shape[0], img_original.shape[1]

        # 3. PREPROCESS: Prepariamo l'immagine per la nostra U-Net (256x256)
        img_input = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_input, (256, 256))
        img_input = img_input / 255.0  # Normalizzazione float32 [0, 1]
        img_input = np.expand_dims(img_input, axis=0)  # Aggiungiamo la dimensione del batch (1, 256, 256, 3)

        # 4. INFERENZA: Prediciamo la maschera
        prediction = model.predict(img_input, verbose=0) # Output shape: (1, 256, 256, 1)
        mask = prediction[0, :, :, 0] # Rimuoviamo il batch e il canale -> (256, 256)
        
        # Applichiamo la soglia (threshold) per rendere la maschera puramente binaria (0 o 255)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Riportiamo la maschera alle dimensioni dell'immagine originale per un confronto perfetto
        binary_mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 5. COSTRUZIONE DEL CONFRONTO (OVERLAY)
        # Creiamo un'immagine completamente rossa delle stesse dimensioni dell'originale
        red_overlay = np.zeros_like(img_original)
        red_overlay[:] = [0, 0, 255] # BGR: Rosso intenso

        # Applichiamo il rosso SOLO dove la maschera binaria ha rilevato una crepa (pixel a 255)
        red_cracks = cv2.bitwise_and(red_overlay, red_overlay, mask=binary_mask_resized)

        # Combiniamo l'immagine originale con le crepe rosse usando la semi-trasparenza
        # alpha = opacità originale (0.7), beta = opacità crepe rosse (0.3)
        comparison_result = cv2.addWeighted(src1=img_original, alpha=0.7, src2=red_cracks, beta=0.3, gamma=0)

        # 6. ENCODING IN MEMORIA: Trasformiamo il risultato in un file scaricabile al volo
        _, buffer = cv2.imencode('.jpg', comparison_result)
        io_buf = io.BytesIO(buffer)

        print("Overlay generato con successo! Invio del file immagine...")
        # Restituiamo direttamente il file binario dell'immagine con il giusto MIME type
        return send_file(io_buf, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Errore durante l'elaborazione dell'immagine: {str(e)}"
        }), 500



'''
Come testarlo su Postman

Per vedere il confronto grafico direttamente dentro Postman, configura la richiesta in questo modo:

Imposta il metodo su POST.

Inserisci l'URL: http://localhost:5000/predict/comparison-overlay.

Vai nella scheda Body, seleziona form-data.

Nella colonna KEY, scrivi image e cambia il tipo da Text a File (appare un piccolo menu a tendina passando il mouse sulla cella).
'''