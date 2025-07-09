import pandas as pd

def riempi_cabinregion(combined_df):
    train_df = combined_df[combined_df['IsTrain'] == True].copy()

    for i in range(1, 8):
        col_name = f'Cabin_region{i}'

        # Assicurati che la colonna sia di tipo Int64 (nullable integer)
        combined_df[col_name] = combined_df[col_name].astype('Int64')
        train_df[col_name] = train_df[col_name].astype('Int64')

        # Conta valori mancanti PRIMA
        mancanti_prima = combined_df[col_name].isna().sum()

        # Costruisci la mappa Group → Cabin_regionX (moda) solo sul training
        gruppi_multipli = train_df['Group'].value_counts()
        gruppi_validi = gruppi_multipli[gruppi_multipli > 1].index

        cabin_gruppo = (
            train_df[train_df['Group'].isin(gruppi_validi)]
            .dropna(subset=[col_name])
            .groupby('Group')[col_name]
            .agg(lambda x: x.mode()[0])  # moda per gruppo
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

        # Imputazione finale con moda globale (solo train)
        moda_cabin = train_df[col_name].mode()
        if not moda_cabin.empty:
            moda_val = moda_cabin[0]
            combined_df[col_name] = combined_df[col_name].fillna(moda_val).astype('Int64')

        # Conta valori mancanti DOPO
        mancanti_dopo = combined_df[col_name].isna().sum()
        print(f"[{col_name}] Valori mancanti prima: {mancanti_prima} | dopo: {mancanti_dopo}")

    return combined_df
