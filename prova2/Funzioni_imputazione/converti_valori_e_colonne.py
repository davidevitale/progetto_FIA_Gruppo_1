import pandas as pd
import numpy as np

def converti_valori_colonne(train_path, val_path, test_path,
                             output_train_path, output_val_path, output_test_path):
    # === 1. Carica i file ===
    train_df = pd.read_excel(train_path)
    val_df   = pd.read_excel(val_path)
    test_df  = pd.read_csv(test_path)

    # === 2. Aggiungi flag identificativi ===
    train_df['IsTrain'] = True
    train_df['IsValidation'] = False
    train_df['IsTest'] = False

    val_df['IsTrain'] = False
    val_df['IsValidation'] = True
    val_df['IsTest'] = False

    test_df['IsTrain'] = False
    test_df['IsValidation'] = False
    test_df['IsTest'] = True

    # === 3. Unisci i tre dataset ===
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # === 4. Feature Engineering ===
    combined_df['Group'] = combined_df['PassengerId'].str.split('_').str[0].astype(int)
    group_counts = combined_df['Group'].value_counts()
    combined_df['Group_size'] = combined_df['Group'].map(group_counts)

    combined_df[['Deck', 'CabinNum', 'Side']] = combined_df['Cabin'].str.split('/', expand=True)
    combined_df.drop(columns=['Cabin'], inplace=True)
    combined_df['CabinNum'] = pd.to_numeric(combined_df['CabinNum'], errors='coerce')

    conditions = [
        (combined_df['CabinNum'] < 300),
        (combined_df['CabinNum'] >= 300) & (combined_df['CabinNum'] < 600),
        (combined_df['CabinNum'] >= 600) & (combined_df['CabinNum'] < 900),
        (combined_df['CabinNum'] >= 900) & (combined_df['CabinNum'] < 1200),
        (combined_df['CabinNum'] >= 1200) & (combined_df['CabinNum'] < 1500),
        (combined_df['CabinNum'] >= 1500) & (combined_df['CabinNum'] < 1800), 
        (combined_df['CabinNum'] >= 1800)
    ]
    labels = [1, 2, 3, 4, 5, 6, 7]
    combined_df['Cabin_region'] = np.select(conditions, labels, default=pd.NA)

    combined_df['Surname'] = combined_df['Name'].str.split().str[-1]
    combined_df.drop(columns=['Name'], inplace=True)

    spesa_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    combined_df[spesa_cols] = combined_df[spesa_cols].fillna(0)
    combined_df['Expendures'] = combined_df[spesa_cols].sum(axis=1)

    bool_cols = combined_df.select_dtypes(include=['bool']).columns
    combined_df[bool_cols] = combined_df[bool_cols].astype(int)

    # === 5. Ritaglia i dataset finali ===
    new_train = combined_df[combined_df['IsTrain'] == 1].copy()
    new_val   = combined_df[combined_df['IsValidation'] == 1].copy()
    new_test  = combined_df[combined_df['IsTest'] == 1].copy()

    # === 6. Salva i file finali ===
    new_train.to_excel(output_train_path, index=False)
    new_val.to_excel(output_val_path, index=False)
    new_test.to_excel(output_test_path, index=False)

    print("File salvati correttamente.")
    return combined_df
