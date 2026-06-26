import cv2
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

class CrackPredictor:
    """
    Classe per gestire l'inferenza e la visualizzazione delle maschere di previsione
    per singole immagini di crepe, compatibile con il modello U-Net.
    """
    def __init__(self, model_path, img_size=(256, 256), custom_objects=None):
        """
        Inizializza il predittore caricando il modello in memoria.
        """
        self.img_size = img_size
        print(f"Caricamento del modello da {model_path}...")
        
        # Carica il modello Keras gestendo eventuali metriche/loss custom
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Modello caricato con successo!")

    def preprocess_image(self, image_path):
        """
        Pre-elaborazione dell'immagine singola.
        La trasforma nello stesso identico formato richiesto dal dataloader.
        """
        # 1. Carica l'immagine con OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Impossibile trovare l'immagine al percorso: {image_path}")
            
        # 2. Converte da BGR (standard OpenCV) a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Salva la dimensione originale (altezza, larghezza) per utilizzi futuri
        orig_shape = img.shape[:2] 
        
        # 3. Ridimensiona alla dimensione della U-Net (es. 256x256)
        img_resized = cv2.resize(img, self.img_size)
        
        # 4. Normalizza i pixel tra 0.0 e 1.0 e converte in float32
        img_normalized = img_resized / 255.0
        img_normalized = img_normalized.astype(np.float32)
        
        # 5. Espande le dimensioni per creare il finto batch richiesto da Keras (1, 256, 256, 3)
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized, orig_shape

    def predict_mask(self, input_batch, threshold=0.5):
        """
        Esecuzione della previsione sulla rete neurale U-Net.
        Ritorna la maschera binaria finale (0 = sfondo, 1 = crepa).
        """
        # Effettua la previsione (mappa di probabilità)
        prediction = self.model.predict(input_batch, verbose=0)
        
        # Rimuove la dimensione del batch (da 1, 256, 256, 1 a 256, 256)
        prediction_mask = prediction[0, :, :, 0]
        
        # Applica la soglia per binarizzare l'immagine (pixel bianchi o neri puri)
        binary_mask = (prediction_mask > threshold).astype(np.uint8)
        
        return binary_mask

    def display_results(self, img_rgb, binary_mask, title="Risultato U-Net"):
        """
        Visualizzazione dei risultati tramite Matplotlib.
        """
        plt.figure(figsize=(12, 6))

        # Subplot 1: Immagine originale ridimensionata
        plt.subplot(1, 2, 1)
        plt.title("Immagine di Input (RGB)")
        plt.imshow(img_rgb)
        plt.axis("off")

        # Subplot 2: Maschera predetta
        plt.subplot(1, 2, 2)
        plt.title("Maschera Predetta (Bianco/Nero)")
        plt.imshow(binary_mask, cmap="gray")
        plt.axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def run_inference(self, image_path, threshold=0.5, show_plot=True):
        """
        Funzione che unisce tutti i passaggi.
        Prende il percorso dell'immagine, fa la pre-elaborazione,
        predice la maschera e opzionalmente mostra il grafico.
        """
        # Esegue i tre passaggi in sequenza
        input_batch, img_rgb, _ = self.preprocess_image(image_path)
        binary_mask = self.predict_mask(input_batch, threshold=threshold)
        
        if show_plot:
            self.display_results(img_rgb, binary_mask)
            
        return binary_mask