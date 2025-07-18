def riempi_deck(combined_df):
    # === IMPUTAZIONE DECK ===
    CD_bef = combined_df['Deck'].isna().sum()

    # Seleziona solo righe del training set
    train_df = combined_df[combined_df['IsTrain'] == 1].copy()

    # Filtra per gruppi con almeno 2 persone
    gruppi_validi = train_df.loc[train_df['Group_size'] > 1, 'Group'].unique()

    # Mantieni solo questi gruppi nel train
    train_df = train_df[train_df['Group'].isin(gruppi_validi)]

    # Calcola la moda del deck per ogni gruppo
    GCD_gb = train_df.groupby(['Group', 'Deck']).size().unstack(fill_value=0)
    deck_mode_per_group = GCD_gb.idxmax(axis=1)

    # Trova indici nel combined_df con Deck mancante e Group con moda nota
    GCD_index = combined_df[
        combined_df['Deck'].isna() & combined_df['Group'].isin(deck_mode_per_group.index)
    ].index

    # Imputa Deck con la moda del gruppo
    combined_df.loc[GCD_index, 'Deck'] = combined_df.loc[GCD_index, 'Group'].map(deck_mode_per_group)

    # Riempie i restanti missing con 'T'
    combined_df['Deck'] = combined_df['Deck'].fillna('T')

    CD_aft = combined_df['Deck'].isna().sum()

    print(f"Deck missing values before: {CD_bef}")
    print(f"Deck missing values after:  {CD_aft}")

    return combined_df


