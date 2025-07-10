import pandas as pd
import numpy as np

def converti_valori_colonne():

    # === 1. Carica i file ===
    train_df = pd.read_excel('C:/Users/dvita/Desktop/TITANIC/train_holdout.xlsx')
    val_df   = pd.read_excel('C:/Users/dvita/Desktop/TITANIC/val_holdout.xlsx')
    test_df  = pd.read_csv('C:/Users/dvita/Desktop/TITANIC/test.csv')

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


    # Group
    combined_df['Group'] = combined_df['PassengerId'].str.split('_').str[0].astype(int)

    # Group size solo su train
    group_counts = combined_df['Group'].value_counts()
    combined_df['Group_size'] = combined_df['Group'].map(group_counts)

    # Split Cabin
    combined_df[['Deck', 'CabinNum', 'Side']] = combined_df['Cabin'].str.split('/', expand=True)
    combined_df.drop(columns=['Cabin'], inplace=True)

    # New features - training set

    combined_df['CabinNum'] = pd.to_numeric(combined_df['CabinNum'], errors='coerce')

    # Crea le colonne Cabin_region1-7 lasciando NaN dove CabinNum è mancante

    conditions = [
    (combined_df['CabinNum'] < 300),
    (combined_df['CabinNum'] >= 300) & (combined_df['CabinNum'] < 600),
    (combined_df['CabinNum'] >= 600) & (combined_df['CabinNum'] < 900),
    (combined_df['CabinNum'] >= 900) & (combined_df['CabinNum'] < 1200),
    (combined_df['CabinNum'] >= 1200) & (combined_df['CabinNum'] < 1500),
    (combined_df['CabinNum'] >= 1500) & (combined_df['CabinNum'] < 1800), 
    (combined_df['CabinNum'] >= 1800)
    ]

    # Etichette corrispondenti
    labels = [1, 2, 3, 4, 5, 6, 7]

    # Crea la nuova colonna con le etichette
    combined_df['Cabin_region'] = np.select(conditions, labels, default=pd.NA)


    # Surname
    combined_df['Surname'] = combined_df['Name'].str.split().str[-1]
    combined_df.drop(columns=['Name'], inplace=True)

    #combined_df['FamilySize'] = combined_df['Surname'].map(lambda x: combined_df['Surname'].value_counts()[x])


    #combined_df[spesa_cols] = combined_df[spesa_cols].fillna(0)
    #combined_df['Expendures'] = combined_df[spesa_cols].sum(axis=1)

    combined_df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    
    # Calcolo spese
    spesa_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    combined_df['Expendures'] = combined_df[spesa_cols].sum(axis=1)

    #combined_df['NoSpending'] = (combined_df['Expendures'] == 0).astype(int)

    bool_cols = combined_df.select_dtypes(include=['bool']).columns
    combined_df[bool_cols] = combined_df[bool_cols].astype(int)

#######################################

#######################################


    # Print dei conteggi per ogni categoria di AgeGroup
    #print("\nValori per ogni categoria di 'AgeGroup':")
    #print(combined_df['AgeGroup'].value_counts(dropna=False))


    #print("Valori mancanti nella colonna AgeGroup:", combined_df['AgeGroup'].isna().sum())


    # === 5. Ritaglia i dataset finali ===
    new_train = combined_df[combined_df['IsTrain']== 1].copy()
    new_val   = combined_df[combined_df['IsValidation']== 1].copy()
    new_test  = combined_df[combined_df['IsTest']== 1].copy()

    # === 6. Salva i file finali ===
    new_train.to_excel('C:/Users/dvita/Desktop/TITANIC/train_df.xlsx', index=False)
    new_val.to_excel('C:/Users/dvita/Desktop/TITANIC/val_df.xlsx', index=False)
    new_test.to_excel('C:/Users/dvita/Desktop/TITANIC/test_df.xlsx', index=False)

    print("File salvati correttamente.")
    return combined_df

