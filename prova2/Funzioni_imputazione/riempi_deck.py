
from collections import Counter

def riempi_deck(combined_df):
    # === IMPUTAZIONE DECK ===
    CD_bef = combined_df['Deck'].isna().sum()

    # Seleziona solo righe del training set
    train_df = combined_df[combined_df['IsTrain'] == True].copy()

    # Calcola il numero di passeggeri per gruppo nel train
    group_counts = train_df['Group'].value_counts()

    # Filtra per gruppi con almeno 2 persone
    valid_groups = group_counts[group_counts >= 2].index

    # Mantieni solo questi gruppi nel train
    train_df = train_df[train_df['Group'].isin(valid_groups)]

    # Calcola la modalità del deck per ogni gruppo
    GCD_gb = train_df.groupby(['Group', 'Deck']).size().unstack(fill_value=0)
    deck_mode_per_group = GCD_gb.idxmax(axis=1)

    # Maschera per missing Deck
    deck_nan_mask = combined_df['Deck'].isna()

    # Solo gruppi con moda calcolata
    group_with_mode = combined_df.loc[deck_nan_mask, 'Group'].isin(deck_mode_per_group.index)

    # Indici da imputare
    GCD_index = combined_df[deck_nan_mask & group_with_mode].index

    # Imputa Deck con la moda
    combined_df.loc[GCD_index, 'Deck'] = combined_df.loc[GCD_index, 'Group'].map(deck_mode_per_group)

    # Riempie i restanti missing con 'T'
    combined_df['Deck'].fillna('T', inplace=True)

    CD_aft = combined_df['Deck'].isna().sum()

    print(f"#Deck missing values before: {CD_bef}")
    print(f"#Deck missing values after:  {CD_aft}")

    return combined_df


