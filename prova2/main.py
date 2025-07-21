from Funzioni_imputazione.converti_valori_e_colonne import converti_valori_colonne
from Funzioni_imputazione.imputazione import imputazione
from Funzioni_imputazione.adaboost import adaboost
from hold import holdout_split
import time

def main():

    # === Input dei percorsi dei file ===
    input_file = input("Inserisci il percorso del file iniziale (.csv): ")
    output_train = input("Inserisci il percorso per salvare il file TRAIN holdout (.xlsx): ")
    output_val = input("Inserisci il percorso per salvare il file VALIDATION holdout (.xlsx): ")
    test_file = input("Inserisci il percorso del file di TEST (.csv): ")

    # === 1) Suddivisione train/validation ===
    holdout_split(
        input_path=input_file,
        train_path=output_train,
        val_path=output_val,
        test_size=0.2,
        stratify_col='Transported',
    )

    # === 2) Conversione e pulizia ===
    output_clean_train = input("Inserisci il percorso per salvare il TRAIN pulito (.xlsx): ")
    output_clean_val = input("Inserisci il percorso per salvare il VALIDATION pulito (.xlsx): ")
    output_clean_test = input("Inserisci il percorso per salvare il TEST pulito (.xlsx): ")

    combined_df = converti_valori_colonne(
        train_path=output_train,
        val_path=output_val,
        test_path=test_file,
        output_train_path=output_clean_train,
        output_val_path=output_clean_val,
        output_test_path=output_clean_test
    )

    # === 3) Imputazione ===
    encoded_train_path = input("Inserisci il percorso per salvare il TRAIN encoded (.xlsx): ")
    encoded_val_path = input("Inserisci il percorso per salvare il VALIDATION encoded (.xlsx): ")
    encoded_test_path = input("Inserisci il percorso per salvare il TEST encoded (.xlsx): ")

    df_train_encoded, df_val_encoded, df_test_encoded = imputazione(
        combined_df,
        output_train=encoded_train_path,
        output_val=encoded_val_path,
        output_test=encoded_test_path
    )

    # === 4) Addestramento, valutazione e creazione submission ===
    submission_filename = input("Inserisci il nome del file di submission da creare (es. submission.csv): ")

    adaboost(
        df_train_encoded=df_train_encoded,
        df_val_encoded=df_val_encoded,
        df_test_encoded=df_test_encoded,
        target_column='Transported',
        id_column='PassengerId',
        submission_filename=submission_filename,
        n_estimators=500,  
    )

    print("Pipeline completata con successo.")

if __name__ == "__main__":
    main()