def riempi_cryo(combined_df):
    # Conta valori mancanti PRIMA
    cryo_nan_before = combined_df['CryoSleep'].isna().sum()

    # Imputazione basata su NoSpending
    combined_df.loc[combined_df['CryoSleep'].isna() & (combined_df['Expendures'] == 0),  'CryoSleep'] = 1
    combined_df.loc[combined_df['CryoSleep'].isna() & (combined_df['Expendures'] > 0), 'CryoSleep'] = 0


    # Conta valori mancanti DOPO
    cryo_nan_after = combined_df['CryoSleep'].isna().sum()

    print(f"[CryoSleep] Valori mancanti prima: {cryo_nan_before}, dopo: {cryo_nan_after}")
    return combined_df

