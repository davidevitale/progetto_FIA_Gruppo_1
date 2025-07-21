def riempi_side(combined_df):
    # Conta valori mancanti PRIMA
    side_nan_before = combined_df['Side'].isna().sum()

    # Calcola la moda solo sul training set
    train_df = combined_df[combined_df['IsTrain'] == 1]
    moda_side = train_df['Side'].mode()

    # Se la moda esiste, usa quel valore per riempire i NaN
    if not moda_side.empty:
        moda_val = moda_side[0]
        combined_df['Side'] = combined_df['Side'].fillna(moda_val)

    # Conta valori mancanti DOPO
    side_nan_after = combined_df['Side'].isna().sum()

    print(f"Valori mancanti in 'Side' prima: {side_nan_before}")
    print(f"Valori mancanti in 'Side' dopo:  {side_nan_after}")
    print(f"Valori Side riempiti: {side_nan_before - side_nan_after}")
    
    return combined_df
