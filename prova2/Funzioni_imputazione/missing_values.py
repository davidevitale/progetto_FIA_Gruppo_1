from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def knn_impute(combined_df, output_train, output_val, output_test):
    exclude_cols = ["IsTrain", "IsValidation", "IsTest", "Transported", "PassengerId"]


    # Converti pd.NA -> np.nan
    combined_df = combined_df.replace({pd.NA: np.nan})

    # Seleziona colonne numeriche
    numeric_cols = combined_df.drop(columns=exclude_cols).select_dtypes(include=[np.number]).columns.tolist()

    # Standardizza solo le colonne numeriche
    scaler = StandardScaler()
    df_numeric = combined_df[numeric_cols]

    # Fit solo sul train set
    train_mask = combined_df['IsTrain'] == 1
    scaler.fit(df_numeric[train_mask])

    df_scaled = df_numeric.copy()
    df_scaled[numeric_cols] = scaler.transform(df_numeric)

    # Imputazione KNN
    imputer = KNNImputer(n_neighbors=5)
    imputed_values = imputer.fit_transform(df_scaled)

    # Ricrea DataFrame imputato
    df_imputed = pd.DataFrame(imputed_values, columns=numeric_cols, index=combined_df.index)

    # De-standardizza
    df_imputed[numeric_cols] = scaler.inverse_transform(df_imputed[numeric_cols])

    # Aggiungi colonne non numeriche
    for col in exclude_cols:
        df_imputed[col] = combined_df[col]

    #Rendi interi i dati (approssimazione)
    df_imputed[numeric_cols] = df_imputed[numeric_cols].round().astype(np.int64)

    # Suddividi i dataset
    df_train_encoded = df_imputed[df_imputed["IsTrain"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_val_encoded   = df_imputed[df_imputed["IsValidation"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])
    df_test_encoded  = df_imputed[df_imputed["IsTest"] == 1].drop(columns=["IsTrain", "IsValidation", "IsTest"])

    # Salva su Excel
    df_train_encoded.to_excel(output_train, index=False)
    df_val_encoded.to_excel(output_val, index=False)
    df_test_encoded.to_excel(output_test, index=False)

    print("File salvati correttamente:")
    print(f"- Train: {output_train}")
    print(f"- Val:   {output_val}")
    print(f"- Test:  {output_test}")

    return df_train_encoded, df_val_encoded, df_test_encoded

