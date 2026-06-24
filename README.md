 # Crack500 - Image Segmentation

Image Segmentation del dataset [Crack500](https://www.kaggle.com/datasets/vangiap/crack500-dataset),
che si concentra sulla ricerca di crepe su una superficie
utilizzando un approccio di deep learning basato sulla U-net

## Obiettivo del progetto

L’obiettivo del progetto è sviluppare un modello di image segmentation binaria in grado di:

- Identificare automaticamente la presenza di crepe nelle immagini
- Generare una maschera pixel-wise (crack / background)
- Utilizzare una rete neurale convoluzionale (U-Net) per il task di segmentazione

## Importazione del Dataset

Per importare il dataset Crack500 è necessario eseguire diversi passaggi:
1. Scaricarne il file .zip da  [kaggle](https://www.kaggle.com/datasets/vangiap/crack500-dataset)
2. Unzippare il file e rinominare la directory unzippata da "archive" a "data".
La strutttura della cartella dev'essere la seguente:
    ```
    data/
    ├── train/
    │   ├── images/
    │   ├── masks/
    │
    ├── val/
    │   ├── images/
    │   ├── masks/
    │
    ├── test/
    │   ├── images/
    │   ├── masks/
    ```
3. Inserire la cartella "data" nel progetto su un editor (esempio PyCharm o VSCode)

Nel "main" è possibile testare il corretto inserimento del dataset


## Fasi del progetto

### 1. Preprocessing

Il preprocessing ha lo scopo di uniformare immagini e maschere
per renderle compatibili con l’architettura della rete (U-Net).

Le operazioni effettuate sul dataset Crack500 sono:

**Immagini**
- Resize a 256x256
- Conversione BGR → RGB
- Normalizzazione in range [0, 1]

**Maschere**
- Resize a 256x256
- Conversione in scala di grigi
- Binarizzazione:
  - 0 = background
  - 1 = crack

### 2. Data Augmentation

Avendo un dataset di dimensione limitata, è utile nel nostro caso applicare data augmentation

Questa tecnica permette di aumentare artificialmente la quantità e la variabilità del dataset di training,
al fine di migliorare la capacità di generalizzazione del modello.

L’obiettivo è rendere il modello robusto a variazioni realistiche delle immagini,
evitando l’overfitting su pattern troppo specifici del dataset originale.

### 3. Creazione della u-net

U-net è tipologia di rete neurale convoluzionale utilizzata nel nostro progetto,
in quanto è progettata specificamente per problemi di image segmentation.


La sua architettura è costituita da un encoder e da un decoder,
e l'ultimo layer della rete produce una mappa di probabilità per ogni pixel,
che viene poi trasformata in una maschera binaria (crack / background)

### 4. Definizione delle metriche e valutazione

La fase di valutazione del modello si basa sia su funzioni di loss utilizzate durante l'addestramento
sia su metriche di valutazione utilizzate per misurare la qualità della segmentazione.

- **Binary Cross Entropy (BCE Loss)**  
  Misura la differenza tra la probabilità predetta dal modello e la classe reale di ciascun pixel

- **Dice Coefficient**  
  Misura il grado di sovrapposizione tra la segmentazione predetta e la ground truth. Assume valori compresi tra 0 e 1, dove 1 indica una corrispondenza perfetta.


## Avvio dell'applicazione

### Requisiti

Prima di avviare l'applicazione, assicurati di avere installato tutte le dipendenze necessarie. Puoi trovarle nel file `requirements.txt`.

Per installarle eseguire il comando:
```bash
pip install -r requirements.txt
```

### Avvio in locale

Per visualizzare l'API in locale è necessario recarsi al seguente indirizzo:

   ```
   http://127.0.0.1:5000/
   ```
## Avvio con Docker

### Prerequisiti

Assicurati di avere installato Docker. Puoi scaricare l'applicazione [qui](https://www.docker.com/products/docker-desktop/).

Verifica l’installazione con:

```bash
docker --version
```

### Utilizzo di Docker

1. Costruisci l'immagine Docker da linea di comando:
   ```bash
   docker build -t crack500 .
   ```

2. Si possono controllare le informazioni dell'immagine appena creata con il comando:
   ```bash
   docker image ls
   ```

3. Costruisci e avvia il container da linea di comando (copiando questo comando, verrà chiamato "seeds"):
   ```bash
   docker run --name progetto-crack500 -p 5000:5000 crack500
   ```
   Se l'operazione è andata a buon fine è possibile vedere lo stavo attivo del container tramite il comando:
   ```bash
   docker ps
   ```

4. Accedi all'applicazione (nella sua route home) tramite il tuo browser all'indirizzo:
   ```
   http://127.0.0.1:5000/
   ```

## API Endpoints

Il progetto è provvisto di diversi endpoint, consultabili nella route **home**.
```
http://127.0.0.1:5000/
```
### dataset/build [POST]

Prima route da dover lanciare. Permette il **caricamento del dataset**.

```
http://127.0.0.1:5000/dataset/build
```

### dataset/augment [POST]

Applica la **data augmentation** al dataset Crack500 precedentemente caricato.

```
http://127.0.0.1:5000/dataset/augment
```

### dataset/preview_aug [GET]

Permette di visionare lo stato dell'augmentation.
Ritorna errore se essa non è nìancora stata effettuata.

```
http://127.0.0.1:5000/dataset/preview_aug
```