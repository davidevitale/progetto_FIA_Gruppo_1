import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def encoding_statico(combined_df):
    """
    Applica Ordinal Encoding alla colonna Age_group e One-Hot Encoding alle altre colonne categoriali.
    Divide il dataset in train, val e test e salva i file in formato Excel.
    """

    print("Colonne iniziali:", combined_df.columns)

    # === Percorsi output ===
    output_train = 'C:/Users/dvita/Desktop/TITANIC/train_encoded.xlsx'
    output_val = 'C:/Users/dvita/Desktop/TITANIC/val_encoded.xlsx'
    output_test = 'C:/Users/dvita/Desktop/TITANIC/test_encoded.xlsx'

    # Ordine logico dei gruppi di età
    ordine_eta = [['0-18', '19-25', '25+']]

    # Applica Ordinal Encoding alla colonna Age_group
    encoder = OrdinalEncoder(categories=ordine_eta)
    combined_df['AgeGroup'] = encoder.fit_transform(combined_df[['AgeGroup']])

    # Colonne da codificare con One-Hot Encoding (escludendo Age_group)
    cols_to_encode = ['HomePlanet', 'Deck', 'Side', 'Destination']

    # One-Hot Encoding con drop_first=True per evitare multicollinearità
    df_encoded = pd.get_dummies(combined_df[cols_to_encode], drop_first=True)

    # Rimuovo le colonne originali codificate
    df_rest = combined_df.drop(columns=cols_to_encode + ['AgeGroup'])

    df_final = pd.concat([df_rest, df_encoded, combined_df[['AgeGroup']]], axis=1)


    # === Estrai i dataset codificati ===
    df_train_encoded = df_final[df_final['IsTrain'] == True].drop(columns=['IsTrain', 'IsValidation', 'IsTest'])
    df_val_encoded = df_final[df_final['IsValidation'] == True].drop(columns=['IsTrain', 'IsValidation', 'IsTest'])
    df_test_encoded = df_final[df_final['IsTest'] == True].drop(columns=['IsTrain', 'IsValidation', 'IsTest'])

    # === Salva i dataset in Excel ===
    df_train_encoded.to_excel(output_train, index=False)
    df_val_encoded.to_excel(output_val, index=False)
    df_test_encoded.to_excel(output_test, index=False)

    print(f"Train codificato salvato in: {output_train}")
    print(f"Val codificato salvato in:   {output_val}")
    print(f"Test codificato salvato in:  {output_test}")

    return df_train_encoded, df_val_encoded, df_test_encoded
