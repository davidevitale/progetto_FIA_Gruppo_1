def riempi_cryo(combined_df):
    # Conta valori mancanti PRIMA
    cryo_nan_before = combined_df['CryoSleep'].isna().sum()

    # Imputazione basata su NoSpending
    combined_df.loc[combined_df['CryoSleep'].isna() & (combined_df['NoSpending'] == True),  'CryoSleep'] = 1

    # Forza valori coerenti in CryoSleep
    combined_df['CryoSleep'] = combined_df['CryoSleep'].map({True: 1, False: 0})  # oppure usa .astype(bool)

    # Conta valori mancanti DOPO
    cryo_nan_after = combined_df['CryoSleep'].isna().sum()

    print(f"[CryoSleep] Valori mancanti prima: {cryo_nan_before}, dopo: {cryo_nan_after}")
    return combined_df

