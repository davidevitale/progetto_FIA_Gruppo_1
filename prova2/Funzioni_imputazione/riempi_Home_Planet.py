import pandas as pd

def riempi_Home_Planet(combined_df):
    if 'Group' not in combined_df.columns:
        combined_df['Group'] = combined_df['PassengerId'].str.split('_').str[0].astype(int)

    if 'Surname' not in combined_df.columns:
        combined_df['Surname'] = combined_df['Name'].str.split().str[-1]

    # Conta valori mancanti PRIMA
    mancanti_prima = combined_df['HomePlanet'].isna().sum()

    # === 1. IMPUTAZIONE PER GRUPPO ===
    # --- Separa il training set ---
    train_df = combined_df[combined_df['IsTrain'] == True].copy()

# --- Costruisci la mappa Group → HomePlanet SOLO sul training ---
    gruppi_multipli = train_df['Group'].value_counts()
    gruppi_validi = gruppi_multipli[gruppi_multipli > 1].index

    homeplanet_gruppo = (
        train_df[train_df['Group'].isin(gruppi_validi)]
        .dropna(subset=['HomePlanet'])
        .groupby('Group')['HomePlanet']
        .agg(lambda x: x.mode()[0])
        .dropna()
        .to_dict()
    )

    # --- Applica la mappa a tutto il dataset ---
    combined_df['HomePlanet'] = combined_df.apply(
        lambda row: homeplanet_gruppo.get(row['Group'], row['HomePlanet'])
        if pd.isna(row['HomePlanet']) else row['HomePlanet'],
        axis=1
    )


    # === 2. IMPUTAZIONE PER DECK ===
    cond_deck = combined_df['HomePlanet'].isna()
    combined_df.loc[cond_deck & (combined_df['Deck'] == 'G'), 'HomePlanet'] = 'Earth'
    combined_df.loc[cond_deck & (combined_df['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'

    # === 3. IMPUTAZIONE PER COGNOME ===

# Costruisci la mappa cognome → HomePlanet solo dal train
    homeplanet_surname = (
        train_df.dropna(subset=['HomePlanet'])
        .groupby('Surname')['HomePlanet']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .dropna()
        .to_dict()
    )

    # Applica la mappa a tutto combined_df
    combined_df['HomePlanet'] = combined_df.apply(
        lambda row: homeplanet_surname.get(row['Surname'], row['HomePlanet'])
        if pd.isna(row['HomePlanet']) else row['HomePlanet'],
        axis=1
    )

        # === 4. IMPUTAZIONE FINALE CON MODA SU TRAIN ===
    moda_homeplanet = train_df['HomePlanet'].mode()
    if not moda_homeplanet.empty:
        moda_hp = moda_homeplanet[0]
        combined_df['HomePlanet'] = combined_df['HomePlanet'].fillna(moda_hp)

    # Conta valori mancanti DOPO
    mancanti_dopo = combined_df['HomePlanet'].isna().sum()

    print(f"Valori mancanti in 'HomePlanet' prima: {mancanti_prima}")
    print(f"Valori mancanti in 'HomePlanet' dopo:  {mancanti_dopo}")
    print(f"Valori HomePlanet riempiti: {mancanti_prima - mancanti_dopo}")

    return combined_df