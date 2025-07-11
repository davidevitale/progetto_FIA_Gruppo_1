from Funzioni_imputazione.converti_valori_e_colonne import converti_valori_colonne
from Funzioni_imputazione.imputazione import imputazione
from hold import holdout_split
from Funzioni_imputazione.adaboost import adaboost

def main():
    # === Input dinamici per i percorsi dei file ===
    excel_file = input("Inserisci il percorso del file iniziale (.xlsx): ")
    output_train = input("Inserisci il percorso per salvare il file TRAIN holdout (.xlsx): ")
    output_val = input("Inserisci il percorso per salvare il file VALIDATION holdout (.xlsx): ")
    test_file = input("Inserisci il percorso del file di TEST (.csv): ")

    
    # === 1) Split train/val da file iniziale ===
    holdout_split(
        input_path=excel_file,
        train_path=output_train,
        val_path=output_val,
        test_size=0.2,
        stratify_col='Transported',
        random_state=42
    )

    output_clean_train = input("Inserisci il percorso per salvare il TRAIN pulito(.xlsx): ")
    output_clean_val = input("Inserisci il percorso per salvare il VALIDATION pulito (.xlsx): ")
    output_clean_test = input("Inserisci il percorso per salvare il TEST pulito (.xlsx): ")

    # === 2) Pulizia e conversione colonne, ritorna combined_df ===
    combined_df = converti_valori_colonne(
        train_path=output_train,
        val_path=output_val,
        test_path=test_file,
        output_train_path=output_clean_train,
        output_val_path=output_clean_val,
        output_test_path=output_clean_test
    )

    encoded_train_path = input("Inserisci il percorso per salvare il TRAIN encoded (.xlsx): ")
    encoded_val_path = input("Inserisci il percorso per salvare il VALIDATION encoded (.xlsx): ")
    encoded_test_path = input("Inserisci il percorso per salvare il TEST encoded (.xlsx): ")

    # === 3) Imputazione sul combined_df ===
    df_train_encoded, df_val_encoded, df_test_encoded = imputazione(
        combined_df,
        output_train=encoded_train_path,
        output_val=encoded_val_path,
        output_test=encoded_test_path
    )

    # === 4) Addestramento modello AdaBoost ===
    adaboost(
        df_train_encoded,
        df_val_encoded,
        target_column='Transported',
        n_estimators=250,
        random_state=42
    )

    print("Imputazione completata.")

if __name__ == "__main__":
    main()
    print("Esecuzione completata con successo.")
