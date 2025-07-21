def riempi_cryo(combined_df):
    # Conta valori mancanti PRIMA
    cryo_nan_before = combined_df['CryoSleep'].isna().sum()

    # Imputazione basata su Expenditures
    combined_df.loc[combined_df['CryoSleep'].isna() & (combined_df['Expenditures'] == 0),  'CryoSleep'] = 1
    combined_df.loc[combined_df['CryoSleep'].isna() & (combined_df['Expenditures'] > 0), 'CryoSleep'] = 0

    # Conta valori mancanti DOPO
    cryo_nan_after = combined_df['CryoSleep'].isna().sum()

    print(f"Valori mancanti in 'CryoSleep' prima: {cryo_nan_before}")
    print(f"Valori mancanti in 'CryoSleep' dopo:  {cryo_nan_after}")
    print(f"Valori CryoSleep riempiti: {cryo_nan_before - cryo_nan_after}")
    
    return combined_df


