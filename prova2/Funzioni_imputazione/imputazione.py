from Funzioni_imputazione.riempi_Home_Planet import riempi_Home_Planet
from Funzioni_imputazione.riempi_cryo import riempi_cryo
from Funzioni_imputazione.riempi_deck import riempi_deck
from Funzioni_imputazione.riempi_Destination import riempi_Destination
from Funzioni_imputazione.riempi_Surname import riempi_Surname
from Funzioni_imputazione.knn_impute import knn_impute
from Funzioni_imputazione.encoding import encoding
from Funzioni_imputazione.riempi_cabinregion import riempi_cabinregion

def imputazione(combined_df, output_train, output_val, output_test):
    combined_df = riempi_Home_Planet(combined_df)
    combined_df = riempi_cryo(combined_df)
    combined_df = riempi_deck(combined_df)
    combined_df = riempi_cabinregion(combined_df)
    combined_df = riempi_Surname(combined_df)
    combined_df = riempi_Destination(combined_df)
    combined_df = encoding(combined_df)

    df_train_encoded, df_val_encoded, df_test_encoded = knn_impute(
        combined_df,
        output_train,
        output_val,
        output_test,
    )

    return df_train_encoded, df_val_encoded, df_test_encoded

