import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

def missing_values(combined_df, output_train, output_val, output_test):

    combined_df = combined_df.drop(columns=['Surname', 'Group'])

    # Converti tutti i booleani in 1/0 tranne la colonna 'Side'
    bool_cols = combined_df.select_dtypes(include=['bool']).columns.tolist()
    if 'Side' in bool_cols:
        bool_cols.remove('Side')

    combined_df[bool_cols] = combined_df[bool_cols].replace({True: 1, False: 0})


    # 2. Definisci le colonne da codificare
    ordinal = ['Group_size']
    categorical = ['Deck', 'HomePlanet', 'Destination', 'Side'] + [f'Cabin_region{i}' for i in range(1, 8)]
    columns_to_encode = ordinal + categorical

    # 3. Fitta l'OrdinalEncoder solo sul train
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    train_mask = combined_df['IsTrain'] == 1
    encoder.fit(combined_df.loc[train_mask, columns_to_encode])

    # 4. Applica encoding su tutto combined_df
    combined_df[columns_to_encode] = encoder.transform(combined_df[columns_to_encode])

    # 5. Uniforma i NaN
    combined_df = combined_df.fillna(np.nan)

    # 6. KNN Imputation: fit solo sul train
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(combined_df.loc[train_mask].drop(columns=["IsTrain", "IsValidation", "IsTest"]))

    # 7. Trasforma tutto il dataframe (esclusi flag)
    features = combined_df.drop(columns=["IsTrain", "IsValidation", "IsTest"])
    imputed_values = imputer.transform(features)

    # 8. Ricostruisci il dataframe completo con i flag originali
    combined_df_imputed = pd.DataFrame(imputed_values, columns=features.columns, index=combined_df.index)
    combined_df_imputed[["IsTrain", "IsValidation", "IsTest"]] = combined_df[["IsTrain", "IsValidation", "IsTest"]]

    # 9. Suddividi i dataset
    df_train_encoded = combined_df_imputed[combined_df_imputed['IsTrain'] == 1]
    df_val_encoded   = combined_df_imputed[combined_df_imputed['IsValidation'] == 1]
    df_test_encoded  = combined_df_imputed[combined_df_imputed['IsTest'] == 1]

    # 10. Salva i dataset
    df_train_encoded.to_excel(output_train, index=False)
    df_val_encoded.to_excel(output_val, index=False)
    df_test_encoded.to_excel(output_test, index=False)

    print("\nFile salvati correttamente:")
    print(f"   Train codificato -> {output_train}")
    print(f"   Val codificato   -> {output_val}")
    print(f"   Test codificato  -> {output_test}")

    return df_train_encoded, df_val_encoded, df_test_encoded