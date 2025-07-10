import pandas as pd

def riempi_Destination(combined_df):
    # Conta valori mancanti PRIMA
    mancanti_prima = combined_df['Destination'].isna().sum()

    # === 1. IMPUTAZIONE PER COGNOME ===

    # Mappa cognome → Destination più frequente
    train_df = combined_df[combined_df['IsTrain'] == 1].copy()

    destination_surname = (
        train_df.dropna(subset=['Destination'])
        .groupby('Surname')['Destination']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .dropna()
        .to_dict()
    )

    combined_df['Destination'] = combined_df.apply(
        lambda row: destination_surname.get(row['Surname'], row['Destination'])
        if pd.isna(row['Destination']) else row['Destination'],
        axis=1
    )


    # === 2. IMPUTAZIONE PER HomePlanet = Mars ===
    cond_mars = (combined_df['Destination'].isna()) & (combined_df['HomePlanet'] == 'Mars')
    combined_df.loc[cond_mars, 'Destination'] = 'TRAPPIST-1e'

    # === 3. IMPUTAZIONE PER HomePlanet = Earth ===
    cond_earth = (combined_df['Destination'].isna()) & (combined_df['HomePlanet'] == 'Earth')
    combined_df.loc[cond_earth, 'Destination'] = 'TRAPPIST-1e'

    # Conta valori mancanti DOPO
    mancanti_dopo = combined_df['Destination'].isna().sum()

    # Stampa risultati
    print(f"Valori mancanti in 'Destination' prima: {mancanti_prima}")
    print(f"Valori mancanti in 'Destination' dopo:  {mancanti_dopo}")
    print(f"Valori Destination riempiti: {mancanti_prima - mancanti_dopo}")

    return combined_df