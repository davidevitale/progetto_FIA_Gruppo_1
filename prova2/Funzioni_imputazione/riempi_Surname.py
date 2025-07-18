import pandas as pd

def riempi_Surname(combined_df):
    # 1. Se manca la colonna Group o Surname, la crea
    if 'Group' not in combined_df.columns:
        combined_df['Group'] = combined_df['PassengerId'].str.split('_').str[0].astype(int)

    if 'Surname' not in combined_df.columns:
        combined_df['Surname'] = combined_df['Name'].str.split().str[-1]

    # Conta valori mancanti PRIMA
    mancanti_prima = combined_df['Surname'].isna().sum()

    train_df = combined_df[combined_df['IsTrain'] == 1].copy()

    # === IMPUTAZIONE PER GRUPPO ===
    # Considera solo gruppi con più di una persona
    gruppi_validi = train_df.loc[train_df['Group_size'] > 1, 'Group'].unique()

    # Costruisce mappa: Group → Surname
    surname_gruppo = (
        train_df[train_df['Group'].isin(gruppi_validi)]
        .dropna(subset=['Surname'])
        .groupby('Group')['Surname']
        .agg(lambda x: x.mode()[0] if x.nunique() == 1 else None)
        .dropna()
        .to_dict()
    )

    # Applica l'imputazione
    combined_df['Surname'] = combined_df.apply(
        lambda row: surname_gruppo.get(row['Group'], None)
        if pd.isna(row['Surname']) else row['Surname'],
        axis=1
    )

    # Conta valori mancanti DOPO
    mancanti_dopo = combined_df['Surname'].isna().sum()

    # Stampa risultato
    print(f"Valori mancanti in 'Surname' prima: {mancanti_prima}")
    print(f"Valori mancanti in 'Surname' dopo:  {mancanti_dopo}")
    print(f"Valori Surname riempiti: {mancanti_prima - mancanti_dopo}")

    return combined_df