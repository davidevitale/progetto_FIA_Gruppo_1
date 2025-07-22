# Spaceship Titanic - Machine Learning Pipeline

Questo progetto affronta la competizione [Spaceship Titanic](https://www.kaggle.com/competitions/titanic) su Kaggle, il cui obiettivo è predire se un passeggero è stato trasportato in un'altra dimensione (`Transported`) sulla base di variabili anagrafiche, comportamentali e logistiche.

Il progetto implementa una pipeline completa di preprocessing, imputazione, codifica e classificazione, con l'obiettivo di ottenere un modello accurato, robusto e privo di data leakage.

---

## Struttura del progetto

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
├── converti_valori_e_colonne.py  
├── imputazione.py  
├── adaboost.py  
├── missing_values.py  
├── main.py  

---

## Pipeline operativa

1. **Suddivisione Holdout**  
   - Split del dataset iniziale in train e validation (80/20 stratificato su 'Transported').

2. **Feature Engineering e Pulizia**  
   - Creazione di nuove colonne: Group, Cabin_region, Expenditures, Surname, ecc.  
   - Conversione dei dati per uniformare il formato.

3. **Imputazione valori mancanti**  
   - Surname + Group + Deck → HomePlanet  
   - Expenditures → CryoSleep  
   - Group + outlier → Deck  
   - HomePlanet + Surname + moda → Destination  
   - Group + valori meno frequenti → Cabin_region
   - moda → Side
   - KNN per imputare valori numerici residui.

4. **Codifica**
   - One-hot encoding delle variabili categoriche.

5. **Addestramento e Predizione**  
   - AdaBoost (con 500 estimatori).  
   - Valutazione sul validation set.  
   - Creazione file di submission .csv per Kaggle.

---

## Come eseguire il codice

1. Apri il terminale nella cartella principale del progetto.  
2. Esegui il comando:

   `python main.py`

3. Il programma ti chiederà, passo dopo passo:
   - Percorso del file iniziale (.csv)
   - Output di train e validation (.xlsx)
   - File di test (.csv)
   - Output dei file puliti e trasformati (.xlsx)
   - Output dei file encoded (.xlsx)
   - Nome del file di submission da creare (es: submission.csv)

---

## Output generati

- File di train/validation/test trasformati (.xlsx)  
- File encoded (.xlsx)  
- File di submission .csv pronto per upload su Kaggle  
- Confusion matrix del validation set

---

## Requisiti

packaging==24.2  
pandas==2.2.3  
numpy==2.0.2  
scikit-learn==1.3.0  
openpyxl==3.1.5  
matplotlib==3.9.4  
seaborn==0.12.2  
pyparsing==3.2.1  
contourpy==1.3.0  
cycler==0.12.1  
et_xmlfile==2.0.0  
fonttools==4.55.3  
kiwisolver==1.4.7  
pillow==11.1.0  
python-dateutil==2.9.0.post0  


---

## 🔧 Parametri personalizzabili

| Parametro       | Descrizione                         | Default       |
|-----------------|-------------------------------------|---------------|
| test_size       | Percentuale per validation          | 0.2           |
| n_estimators    | Stimatori del modello AdaBoost      | 500           |
| max_depth       | Profondità albero                   | 3             |
| target_column   | Variabile da predire                | Transported   |
| id_column       | Identificativo per la submission    | PassengerId   |

---

## Conclusioni

Questo progetto ha permesso di costruire una pipeline completa di preprocessing e classificazione per il dataset Spaceship Titanic.

Grazie a un'analisi esplorativa approfondita e all'uso di funzioni modulari, ogni variabile mancante è stata gestita con logiche coerenti e motivabili:

- Le imputazioni si sono basate su informazioni strutturali (Group, Surname, Deck), logiche deduttive (es. spesa → CryoSleep), e metodi statistici (KNN).
- L'utilizzo di grafici (heatmap, countplot, pie chart) ha supportato la costruzione e la validazione di ogni strategia.
- Le codifiche categoriali sono state gestite con attenzione al data leakage, e il modello è stato validato separando train/val/test.

I modelli AdaBoost e Random Forest hanno dimostrato buone prestazioni.
In particolare, AdaBoost ha restituito la migliore accuratezza sul validation set.

Il progetto ha rafforzato la comprensione delle buone pratiche in ML, dalla gestione dei dati mancanti alla produzione di un modello solido e pronto per la submission.

---
