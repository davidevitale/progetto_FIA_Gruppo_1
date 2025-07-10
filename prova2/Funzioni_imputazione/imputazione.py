from Funzioni_imputazione.riempi_Home_Planet import riempi_Home_Planet
from Funzioni_imputazione.riempi_cryo import riempi_cryo
from Funzioni_imputazione.riempi_deck import riempi_deck
from Funzioni_imputazione.riempi_Destination import riempi_Destination
from Funzioni_imputazione.riempi_Surname import riempi_Surname
from Funzioni_imputazione.missing_values import knn_impute
from Funzioni_imputazione.encoding import encoding
from Funzioni_imputazione.adaboost import adaboost
from Funzioni_imputazione.riempi_cabinregion import riempi_cabinregion
from Funzioni_imputazione.random_forest import random_forest


def imputazione(combined_df):
    combined_df = riempi_Home_Planet(combined_df)
    combined_df = riempi_cryo(combined_df)
    combined_df = riempi_deck(combined_df)
    combined_df = riempi_cabinregion(combined_df)
    combined_df = riempi_Surname(combined_df)
    combined_df = riempi_Destination(combined_df)
    combined_df = encoding(combined_df)
    
    output_train = 'C:/Users/dvita/Desktop/TITANIC/train_encoded.xlsx'
    output_val = 'C:/Users/dvita/Desktop/TITANIC/val_encoded.xlsx'
    output_test = 'C:/Users/dvita/Desktop/TITANIC/test_encoded.xlsx'

    # Chiamo missing_values e assegno i dataframe ritornati
    df_train_encoded, df_val_encoded, df_test_encoded = knn_impute(combined_df, output_train, output_val, output_test)
    
    # Ora uso df_train_encoded e df_val_encoded per addestrare il modello adaboost
    #model = adaboost(df_train_encoded, df_val_encoded, target_column, n_estimators=250, random_state=42)

    #model = random_forest(df_train_encoded, df_val_encoded, target_column, n_estimators=500, max_depth=4, random_state=42)
    
    return df_train_encoded, df_val_encoded, df_test_encoded
