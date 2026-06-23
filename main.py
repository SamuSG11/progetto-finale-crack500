from it.akron.dataset import Crack500LocalDataLoader
from it.akron.metrics import SegmentationMetrics
from it.akron.unet import UNetStandard
from it.akron.augmentation import CrackDataAugmenter
import tensorflow as tf
import numpy as np




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
    # BLOCCO DI TEST DIAGNOSTICI (Tabulato dentro il main)
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



if __name__ == "__main__":
    main()