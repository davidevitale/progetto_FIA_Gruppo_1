# Spaceship Titanic - Machine Learning Pipeline

Questo progetto affronta la competizione [Spaceship Titanic](https://www.kaggle.com/competitions/titanic) su Kaggle, il cui obiettivo è predire se un passeggero è stato trasportato in un'altra dimensione (`Transported`) sulla base di variabili anagrafiche, comportamentali e logistiche.

Il progetto implementa una pipeline completa di preprocessing, imputazione, codifica e classificazione, con l'obiettivo di ottenere un modello accurato, robusto e privo di data leakage.

---

## Struttura del progetto

```
├── Funzioni_imputazione/
│   ├── riempi_Home_Planet.py
│   ├── riempi_cryo.py
│   ├── riempi_deck.py
│   ├── riempi_Destination.py
│   ├── riempi_Surname.py
│   ├── riempi_cabinregion.py
│   ├── knn_impute.py
│   └── encoding.py
│
├── studio/
│   ├── studio_cryo_e_spending.py
│   ├── studio_Homeplanet_Surname.py
│   ├── studio_Surname_Group.py
│   └── Studio_Vip_group.py
│
├── converti_valori_e_colonne.py
├── imputazione.py
├── preprocessing.py
├── adaboost.py
├── random_forest.py
├── missing_values.py
```

---

## 🔁 Pipeline

### 1. **Caricamento e trasformazione dati**
- `converti_valori_e_colonne.py`  
  Unisce i dataset e costruisce nuove feature (`Group`, `Cabin_region`, `Expenditures`, `Surname`, ecc.).

### 2. **Imputazione valori mancanti**
- `imputazione.py`: applica in ordine:
  - `riempi_Home_Planet.py`: group, deck, surname
  - `riempi_cryo.py`: spesa → CryoSleep
  - `riempi_deck.py`: moda del gruppo
  - `riempi_Destination.py`: cognome e pianeta
  - `riempi_Surname.py`: gruppi coerenti
  - `riempi_cabinregion.py`: ultima lettera della cabina
  - `encoding.py`: One-Hot encoding
  - `knn_impute.py`: imputazione finale numerica via KNN

### 3. **Visualizzazione e analisi dei dati**
- Script nella cartella `studio/` e `preprocessing.py` giustificano ogni strategia con analisi grafiche (heatmap, istogrammi, countplot...).

### 4. **Modellazione e classificazione**
- `adaboost.py`: AdaBoost con DecisionTree
- `random_forest.py`: classificatore RandomForest

---

## Come eseguire

1. Converti e unisci i dati:
```bash
python converti_valori_e_colonne.py
```

2. Prepara il dataset completo:
```bash
python imputazione.py
```

3. Esegui un modello:
```bash
python adaboost.py
# oppure
python random_forest.py
```

---

## Requisiti

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl
- fpdf (opzionale per export PDF)

Installa con:

```bash
pip install -r requirements.txt
```

---

## Conclusioni

Questo progetto ha permesso di costruire una pipeline completa di preprocessing e classificazione per il dataset Spaceship Titanic.

Grazie a un'analisi esplorativa approfondita e all'uso di funzioni modulari, ogni variabile mancante è stata gestita con logiche coerenti e motivabili:

- Le imputazioni si sono basate su **informazioni strutturali** (Group, Surname, Deck), **logiche deduttive** (es. spesa → CryoSleep), e **metodi statistici** (KNN).
- L'utilizzo di grafici (heatmap, countplot, pie chart) ha supportato la costruzione e la validazione di ogni strategia.
- Le codifiche categoriali sono state gestite con attenzione al data leakage, e il modello è stato validato separando train/val/test.

I modelli **AdaBoost** e **Random Forest** hanno dimostrato buone prestazioni. In particolare, AdaBoost ha beneficiato della qualità del preprocessing e ha restituito la migliore accuratezza sul validation set.

Il progetto ha rafforzato la comprensione delle buone pratiche in ML, dalla gestione dei dati mancanti alla produzione di un modello solido e pronto per la submission.

---

## Output

A valle della pipeline, il progetto produce i seguenti file:

- `train_encoded.xlsx` — dataset di training preprocessato, codificato e imputato
- `val_encoded.xlsx` — validation set separato per la valutazione dei modelli
- `test_encoded.xlsx` — dataset di test trasformato, pronto per la predizione
- `submission.csv` — file pronto per la sottomissione su Kaggle, con `PassengerId` e `Transported`

Questi file vengono generati automaticamente dai moduli `imputazione.py`, `knn_impute.py` e dai modelli finali (`adaboost.py`, `random_forest.py`).

---

