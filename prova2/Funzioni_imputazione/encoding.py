
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encoding(combined_df):
    # 1. Seleziona il train set
    train_mask = combined_df['IsTrain'] == 1

    # 3. Rimuovi colonne inutili
    combined_df = combined_df.drop(columns=['Surname', 'Group', 'Expenditures'])

    # 4. Colonne categoriche
    categorical = ['Deck', 'HomePlanet', 'Destination', 'Side', 'Cabin_region']

    # 5. Standardizza i NaN per le categoriche
    combined_df[categorical] = combined_df[categorical].astype('object')

    # 6. OneHotEncoder con gestione valori sconosciuti
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # 7. Fit solo sul train set
    encoder.fit(combined_df.loc[train_mask, categorical])

    # 8. Trasforma tutto il dataset
    encoded_array = encoder.transform(combined_df[categorical])
    encoded_df = pd.DataFrame(encoded_array,
                              columns=encoder.get_feature_names_out(categorical),
                              index=combined_df.index)

    # 9. Rimuovi colonne categoriche originali e unisci le nuove
    combined_df = combined_df.drop(columns=categorical)
    combined_df = pd.concat([combined_df, encoded_df], axis=1)

    # 10. Ritorna il DataFrame codificato e la serie PassengerId del test set
    return combined_df