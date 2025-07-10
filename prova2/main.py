from Funzioni_imputazione.converti_valori_e_colonne import converti_valori_colonne
from Funzioni_imputazione.imputazione import imputazione
from hold import holdout_split
from Funzioni_imputazione.adaboost import adaboost


def main():
    # Percorso file input e output
    excel_file = 'C:/Users/dvita/Desktop/TITANIC/train.xlsx'
    output_train = 'C:/Users/dvita/Desktop/TITANIC/train_holdout.xlsx'
    output_val = 'C:/Users/dvita/Desktop/TITANIC/val_holdout.xlsx'

    # 1) Split train/val da file iniziale
    holdout_split(
        input_path=excel_file, train_path=output_train, val_path=output_val, test_size=0.2, stratify_col='Transported', random_state=42
    )

    # 2) Pulizia e conversione colonne, ritorna combined_df
    combined_df = converti_valori_colonne()

    # 3) Imputazione sul combined_df
    df_train_encoded, df_val_encoded, df_test_encoded = imputazione(combined_df)

    # 4) Addestramento modello AdaBoost
    adaboost(df_train_encoded, df_val_encoded, target_column = 'Transported', n_estimators=250, random_state=42)

    # Puoi salvare df_finale oppure stampare info
    print("Imputazione completata.")

if __name__ == "__main__":
    main()
    print("Esecuzione completata con successo.")



##############
# from Funzioni_imputazione.converti_valori_e_colonne import converti_valori_colonne
#from Funzioni_imputazione.imputazione import imputazione
#from hold import holdout_split

#def main():

    # 1) Split train/val da file iniziale
    #holdout_split()

    # 2) Pulizia e conversione colonne, ritorna combined_df
    #combined_df = converti_valori_colonne()

    # 3) Imputazione sul combined_df
    #combined_df = imputazione(combined_df)

    # Puoi salvare df_finale oppure stampare info
    #print("Imputazione completata.")

#if __name__ == "__main__":
    #main()
    #print("Esecuzione completata con successo.")
    ###############