import os
import json
import io
import matplotlib
# Forziamo matplotlib a usare un backend non interattivo per evitare errori di thread su server web
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class PerformanceVisualizer:
    def __init__(self, models_config):
        """
        Inizializza il visualizzatore con la configurazione dei percorsi dei modelli.
        """
        self.config = models_config

    def _extract_metrics(self, model_key):
        """
        Metodo privato per estrarre Train, Val e Test Dice Coefficient dai JSON di un modello.
        """
        cfg = self.config[model_key]
        train_dice = 0.0
        val_dice = 0.0
        test_dice = 0.0

        # 1. Estrazione da stats (Train e Validation)
        if os.path.exists(cfg["json_stats_path"]):
            with open(cfg["json_stats_path"], "r") as f:
                data = json.load(f)
                # Adattiamo l'estrazione in base alla struttura del tuo JSON
                train_dice = data.get("best_epoch_metrics", {}).get("train", {}).get("dice_coefficient", 0.0)
                val_dice = data.get("best_epoch_metrics", {}).get("validation", {}).get("dice_coefficient", 0.0)
                
                # Se nel tuo JSON le metriche sono sotto "compile_metrics"
                if train_dice == 0.0:
                    train_dice = data.get("best_epoch_metrics", {}).get("train", {}).get("compile_metrics", 0.0)
                if val_dice == 0.0:
                    val_dice = data.get("best_epoch_metrics", {}).get("validation", {}).get("compile_metrics", 0.0)

        # 2. Estrazione da Test Metrics
        if os.path.exists(cfg["json_test_path"]):
            with open(cfg["json_test_path"], "r") as f:
                data = json.load(f)
                test_dice = data.get("test_performance", {}).get("dice_coefficient", 0.0)

                if test_dice == 0.0:
                    test_dice = data.get("test_performance", {}).get("compile_metrics", 0.0)

        return train_dice, val_dice, test_dice

    def generate_comparison_chart(self):
        """
        Genera il Grouped Bar Chart e restituisce un BytesIO buffer contenente l'immagine PNG.
        """
        models_labels = []
        train_scores = []
        val_scores = []
        test_scores = []

        # Raccogliamo i dati per ogni modello registrato
        for model_key in self.config.keys():
            t_dice, v_dice, te_dice = self._extract_metrics(model_key)
            
            # Formattiamo il nome per renderlo carino sul grafico (es. efficientnet_unet -> Efficientnet Unet)
            models_labels.append(model_key.replace("_", " ").title())
            train_scores.append(t_dice)
            val_scores.append(v_dice)
            test_scores.append(te_dice)

        # Costruzione del grafico
        x = np.arange(len(models_labels))
        width = 0.22  # Spessore delle singole barre

        fig, ax = plt.subplots(figsize=(9, 6))

        # Disegniamo le barre raggruppate con una palette moderna ed elegante
        rects1 = ax.bar(x - width, train_scores, width, label='Train', color='#4f46e5')       # Indigo
        rects2 = ax.bar(x, val_scores, width, label='Validation', color='#06b6d4')          # Cyan
        rects3 = ax.bar(x + width, test_scores, width, label='Test', color='#10b981')         # Emerald

        # Configurazione assi e label
        ax.set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Confronto Performance Modelli U-Net', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)  # Spazio extra in alto per le etichette numeriche
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', frameon=True, shadow=False)

        # Funzione interna per inserire i valori sopra le barre
        def _add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0: # Evitiamo di scrivere 0.000 se il dato manca
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='semibold')

        _add_labels(rects1)
        _add_labels(rects2)
        _add_labels(rects3)

        plt.tight_layout()

        # Salviamo il grafico in un buffer di memoria in formato PNG
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        
        # Chiudiamo esplicitamente la figura per liberare la memoria RAM del server
        plt.close(fig)

        return img_buffer