import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def encoding(combined_df):
    # 1. Seleziona il train set
    train_mask = combined_df['IsTrain'] == 1

    # Estrai il PassengerId del test set da combined_df
    passenger_id_test = combined_df.loc[combined_df['IsTest'] == 1, 'PassengerId'].copy()
# Poi fai encoding/imputazione su df_test senza modificarlo


    # 2. Rimuovi colonne inutili
    combined_df = combined_df.drop(columns=['Surname', 'Group', 'PassengerId', 'Expendures'])

    # 3. Colonne categoriche
    categorical = ['Deck', 'HomePlanet', 'Destination', 'Side']

    # 4. Standardizza i NaN
    combined_df[categorical] = combined_df[categorical].astype('object')

    # 5. OrdinalEncoder: unknown_value deve essere un intero
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # 6. Fit solo sul train
    encoder.fit(combined_df.loc[train_mask, categorical])

    # 7. Trasforma tutto il dataset
    combined_df[categorical] = encoder.transform(combined_df[categorical])

    return combined_df