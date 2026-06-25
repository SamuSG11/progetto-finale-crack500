from it.akron.dataset import Crack500LocalDataLoader
from it.akron.metrics import SegmentationMetrics
from it.akron.unet import UNetStandard
from it.akron.augmentation import CrackDataAugmenter
from it.akron.prediction import CrackPredictor

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras



def main():

    # 2. Inizializziamo il nuovo loader locale
    data_loader = Crack500LocalDataLoader(batch_size=16, img_size=(256, 256))

    # Carichiamo i tre set di dati puri
    train_ds = data_loader.get_dataset(split="train", shuffle=True)
    val_ds   = data_loader.get_dataset(split="val", shuffle=False)
    test_ds  = data_loader.get_dataset(split="test", shuffle=False)

    # 3. Applichiamo l'augmentation a parte
    augmenter = CrackDataAugmenter()
    train_ds = augmenter.apply(train_ds)

    # 4. Ottimizzazione finale della memoria
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
# =================================================================
# BLOCCO DI TEST DIAGNOSTICI 
# =================================================================
    print("\n=== AVVIO TEST DI VERIFICA CARICAMENTO ===")
    
    # Estraiamo un solo batch dal train_ds usando .take(1)
    for test_images, test_masks in train_ds.take(1):
        # Convertiamo in array numpy per analizzarli facilmente
        img_np = test_images.numpy()
        mask_np = test_masks.numpy()
        
        print(f"[TEST 1] Dimensioni Batch Immagini: {img_np.shape}") 
        print(f"[TEST 1] Dimensioni Batch Maschere: {mask_np.shape}")
        # Verifica dimensioni: ci aspettiamo (batch_size, altezza, larghezza, canali)
        assert img_np.shape == (16, 256, 256, 3), "Errore nelle dimensioni delle immagini!"
        assert mask_np.shape == (16, 256, 256, 1), "Errore nelle dimensioni delle maschere!"
        print("-> TEST 1 SUPERATO: Dimensioni dei tensori corrette.")
        
        print(f"[TEST 2] Range pixel Immagini: Min={img_np.min():.4f}, Max={img_np.max():.4f}")
        print(f"[TEST 2] Range pixel Maschere: Min={mask_np.min():.4f}, Max={mask_np.max():.4f}")
        # Verifica normalizzazione: i valori devono essere compresi tra 0.0 e 1.0
        assert 0.0 <= img_np.min() and img_np.max() <= 1.0, "Le immagini non sono normalizzate tra [0, 1]!"
        assert 0.0 <= mask_np.min() and mask_np.max() <= 1.0, "Le maschere non sono normalizzate tra [0, 1]!"
        print("-> TEST 2 SUPERATO: Normalizzazione [0, 1] corretta.")
        
        print("[TEST 3] Verifica coerenza Data Augmentation...")
        # Controlliamo che l'augmentation non abbia introdotto valori strani (NaN o infiniti)
        assert not np.isnan(img_np).any(), "Rilevati valori NaN nelle immagini dopo l'augmentation!"
        assert not np.isnan(mask_np).any(), "Rilevati valori NaN nelle maschere dopo l'augmentation!"
        print("-> TEST 3 SUPERATO: L'augmentation ha elaborato il batch senza errori matematici.")

    print("=== TUTTI I TEST SONO STATI SUPERATI CON SUCCESSO! ===\n")

    print("Pipeline dati inizializzata e ottimizzata con successo!")

# =====================================================================
# CARICAMENTO DEL MODELLO
# =====================================================================
    '''
    Il modello U-NET è stato addestrato con successo su un notebook in Google Colab e salvato come "best_unet_model.keras" nella cartella /models.
    '''
    MODEL_PATH = "models/best_unet_model.keras"

    print("Caricamento del modello in corso...")
    # Passiamo le metriche custom tramite il parametro custom_objects
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_bce_loss": SegmentationMetrics.dice_bce_loss,
            "dice_coefficient": SegmentationMetrics.dice_coefficient
        }
    )
    print("Modello caricato con successo!")

    '''
    print("\nAvvio della valutazione su tutto il Test Set...")
    
    # evaluate() restituisce una lista con i valori delle metriche nell'ordine in cui sono state compilate
    results = model.evaluate(test_ds, verbose=1)
    
    # Di solito l'ordine è: [Loss, Accuracy, Dice_Coefficient]
    # Puoi verificare l'ordine esatto stampando model.metrics_names
    metrics_summary = dict(zip(model.metrics_names, results))

    # --- 4. Stampa del Report Finale ---
    print("\n" + "="*40)
    print("        REPORT PERFORMANCE COMPLETO")
    print("="*40)
    for metric_name, value in metrics_summary.items():
        print(f"-> {metric_name:<20}: {value:.4f}")
    print("="*40)
    '''

# ====================================================================
# PREVISIONI SU UNA NUOVA IMMAGINE 
# ====================================================================

    try:
        custom_obj = {
            "dice_bce_loss": SegmentationMetrics.dice_bce_loss,
            "dice_coefficient": SegmentationMetrics.dice_coefficient
        }
    except ImportError:
        custom_obj = {}
    
    # 2. Inizializzi il predittore (carica il modello UNA sola volta)
    predictor = CrackPredictor(
        model_path="models/best_unet_model.keras", 
        img_size=(256, 256),
        custom_objects=custom_obj
    )
    

    # 3. Lanci l'inferenza su qualsiasi immagine inserendo semplicemente il percorso della foto
    PATH_FOTO = "exmples/CRACK500_esempio_immagine.jpg"
    maschera_finale = predictor.run_inference(PATH_FOTO, threshold=0.5, show_plot=True)

    # Ora nella variabile 'maschera_finale' hai l'array NumPy della maschera binaria 
    # che puoi salvare su disco, analizzare o usare per calcolare la percentuale di crepe!


if __name__ == "__main__":
    main()