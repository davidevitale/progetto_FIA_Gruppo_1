
import pandas as pd

def riempi_cabinregion(combined_df):
    train_df = combined_df[combined_df['IsTrain'] == 1].copy()

    col_name = 'Cabin_region'

    # Assicurati che la colonna sia di tipo Int64 (nullable integer)
    combined_df[col_name] = combined_df[col_name].astype('Int64')
    train_df[col_name] = train_df[col_name].astype('Int64')

    mancanti_prima = combined_df[col_name].isna().sum()

    # Costruisci la mappa Group → Cabin_region (moda) solo sul training
    gruppi_validi = train_df.loc[train_df['Group_size'] > 1, 'Group'].unique()

    cabin_gruppo = (
    train_df[train_df['Group'].isin(gruppi_validi)]
    .dropna(subset=[col_name])
    .agg(lambda x: x.mode()[0])
    .dropna()
    .to_dict()
    )

    # Imputazione per gruppo
    def imputa_cabin_region(row):
        if pd.isna(row[col_name]):
            return cabin_gruppo.get(row['Group'], pd.NA)
        else:
            return row[col_name]

    combined_df[col_name] = combined_df.apply(imputa_cabin_region, axis=1)

    # Conta frequenze iniziali solo sul train set (escludendo NaN)
    freq = train_df[col_name].value_counts(dropna=True).to_dict()

    # Prendi l'indice (posizioni) di tutti i NaN nella colonna nel combined_df
    nan_indices = combined_df[combined_df[col_name].isna()].index.tolist()

    for idx in nan_indices:
        if not freq:
            break
        min_cat = min(freq, key=freq.get)
        combined_df.at[idx, col_name] = min_cat
        freq[min_cat] += 1

    combined_df[col_name] = combined_df[col_name].astype('Int64')
    mancanti_dopo = combined_df[col_name].isna().sum()
    print(f"[{col_name}] Valori mancanti prima: {mancanti_prima} | dopo: {mancanti_dopo}")

    return combined_df
