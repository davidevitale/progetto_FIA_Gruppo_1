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

    # Standardizza solo le colonne numeriche
    scaler = StandardScaler()
    df_numeric = combined_df[numeric_cols]

    # Fit solo sul train set
    train_mask = combined_df['IsTrain'] == 1
    scaler.fit(df_numeric[train_mask])

    df_scaled = df_numeric.copy()
    df_scaled[numeric_cols] = scaler.transform(df_numeric)

    df_scaled_full = pd.concat([df_scaled, combined_df[dummy_cols]], axis=1)

    # Imputazione KNN
    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(df_scaled_full)

    # Ricrea DataFrame imputato
    df_imputed = pd.DataFrame(imputed_values, columns=df_scaled_full.columns, index=combined_df.index)

    # De-standardizza
    df_imputed[numeric_cols] = scaler.inverse_transform(df_imputed[numeric_cols])

    negative_mask = (df_imputed[numeric_cols] < 0).any(axis=1)
    n_dropped = negative_mask.sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with negative values in numeric features after de-standardization.")
        df_imputed = df_imputed.drop(df_imputed[negative_mask].index)

    # Aggiungi colonne eliminate dall'imputazione
    for col in exclude_cols:
        df_imputed[col] = combined_df.loc[df_imputed.index, col]

    #Rendi interi i dati (approssimazione)
    df_imputed[numeric_cols] = df_imputed[numeric_cols].round().astype(np.int64)

    spesa_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
     # Ora aggiungi le colonne log per le spese (dopo imputazione)
    for col in spesa_cols:
        df_imputed[f'Log_{col}'] = np.log1p(df_imputed[col])

    # Droppa le colonne originali di spesa, tieni solo le log
    df_imputed = df_imputed.drop(columns=spesa_cols)

    # Suddividi i dataset
    df_train_encoded = df_imputed[df_imputed["IsTrain"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_val_encoded   = df_imputed[df_imputed["IsValidation"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_test_encoded  = df_imputed[df_imputed["IsTest"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])

    # Salva su Excel
    df_train_encoded.to_excel(output_train, index=False)
    df_val_encoded.to_excel(output_val, index=False)
    df_test_encoded.to_excel(output_test, index=False)

    return df_train_encoded, df_val_encoded, df_test_encoded

