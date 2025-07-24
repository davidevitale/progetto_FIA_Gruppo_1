from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def knn_impute(combined_df, output_train, output_val, output_test):
    exclude_cols = ["IsTrain", "IsValidation", "IsTest", "Transported", "PassengerId"]

    # Converti pd.NA -> np.nan
    combined_df = combined_df.replace({pd.NA: np.nan})

        # rileva tutte le colonne con solo valori {0,1} (dopo aver tolto NaN)
    dummy_cols = [
    c for c in combined_df.columns 
    if c != 'Transported'
    and combined_df[c].dropna().isin([0,1]).all() 
    and combined_df[c].dtype in ['int64', 'float64']
    ]

    # escludile dallo scaling
    cols_to_drop = exclude_cols + dummy_cols
    numeric_cols = combined_df.drop(columns=cols_to_drop) \
                            .select_dtypes(include=[np.number]) \
                            .columns.tolist()

    df_numeric = combined_df[numeric_cols].copy()
    mask_nan = df_numeric.isna()  # mask valori NaN originali

    # Salva copia originale df_numeric con NaN (per ricostruzione finale)
    df_numeric_orig = df_numeric.copy()

    # Definisci train mask
    train_mask = combined_df['IsTrain'] == 1

    # Standardizza solo train set per fit
    scaler = StandardScaler()
    scaler.fit(df_numeric.loc[train_mask].dropna())

    # Trasforma tutto df_numeric (NaN rimangono NaN)
    df_numeric_scaled = pd.DataFrame(
        scaler.transform(df_numeric), 
        columns=df_numeric.columns, 
        index=df_numeric.index
    )

    # Combina numeriche standardizzate + dummy cols originali (non standardizzate)
    df_scaled_full = pd.concat([df_numeric_scaled, combined_df[dummy_cols]], axis=1)

    # Imputazione KNN su dataset combinato
    imputer = KNNImputer(n_neighbors=15)
    imputed_array = imputer.fit_transform(df_scaled_full)
    df_imputed_scaled = pd.DataFrame(imputed_array, columns=df_scaled_full.columns, index=df_scaled_full.index)

    # Ora de-standardizza SOLO i valori imputati nella parte numerica
    df_numeric_imputed = df_numeric_orig.copy()  # base: valori originali

    for col in numeric_cols:
        # Riga per cui valore imputato (dove mask_nan==True)
        rows_to_destandardize = mask_nan[col]
        if rows_to_destandardize.any():
            # Prendi i valori imputati standardizzati
            vals_scaled = df_imputed_scaled.loc[rows_to_destandardize, col].values.reshape(-1, 1)
            # De-standardizza
            vals_orig_scale = scaler.inverse_transform(
                np.column_stack([vals_scaled if c == col else np.zeros_like(vals_scaled) for c in numeric_cols])
            )[:, numeric_cols.index(col)]
            # Inserisci i valori de-standardizzati solo nelle posizioni imputate
            df_numeric_imputed.loc[rows_to_destandardize, col] = vals_orig_scale

    # Ricostruisci il dataset finale con numeriche aggiornate e dummy originali
    df_final = pd.concat([df_numeric_imputed, combined_df[dummy_cols]], axis=1)

    # Stampa righe con almeno un valore negativo nelle colonne numeriche (imputate)
    num_rows_with_neg_after = (df_final < 0).any(axis=1).sum()
    print(f"Numero di righe con almeno un valore negativo dopo imputazione: {num_rows_with_neg_after}")


    # Mantieni le colonne escluse così come sono
    for col in exclude_cols:
        if col in combined_df.columns:
            df_final[col] = combined_df[col]

    #Rendi interi i dati (approssimazione)
    df_final[numeric_cols] = df_final[numeric_cols].round().astype(np.int64)

    # Suddividi i dataset
    df_train_encoded = df_final[df_final["IsTrain"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_val_encoded   = df_final[df_final["IsValidation"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_test_encoded  = df_final[df_final["IsTest"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])

    # Salva su Excel
    df_train_encoded.to_excel(output_train, index=False)
    df_val_encoded.to_excel(output_val, index=False)
    df_test_encoded.to_excel(output_test, index=False)

    return df_train_encoded, df_val_encoded, df_test_encoded

