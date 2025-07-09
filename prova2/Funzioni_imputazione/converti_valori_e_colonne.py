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
    combined_df['Cabin_region1'] = combined_df['CabinNum'].apply(lambda x: 1 if x < 300 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region2'] = combined_df['CabinNum'].apply(lambda x: 1 if 300 <= x < 600 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region3'] = combined_df['CabinNum'].apply(lambda x: 1 if 600 <= x < 900 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region4'] = combined_df['CabinNum'].apply(lambda x: 1 if 900 <= x < 1200 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region5'] = combined_df['CabinNum'].apply(lambda x: 1 if 1200 <= x < 1500 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region6'] = combined_df['CabinNum'].apply(lambda x: 1 if 1500 <= x < 1800 else (0 if pd.notna(x) else pd.NA))
    combined_df['Cabin_region7'] = combined_df['CabinNum'].apply(lambda x: 1 if x >= 1800 else (0 if pd.notna(x) else pd.NA))


    # Surname
    combined_df['Surname'] = combined_df['Name'].str.split().str[-1]
    combined_df.drop(columns=['Name'], inplace=True)

    #combined_df['FamilySize'] = combined_df['Surname'].map(lambda x: combined_df['Surname'].value_counts()[x])


    #combined_df[spesa_cols] = combined_df[spesa_cols].fillna(0)
    #combined_df['Expendures'] = combined_df[spesa_cols].sum(axis=1)

    combined_df[['VIP', 'CryoSleep', 'RoomService' 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = combined_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    
        # Calcolo spese
    spesa_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    combined_df['Expendures'] = combined_df[spesa_cols].sum(axis=1)

    combined_df['NoSpending'] = (combined_df['Expendures'] == 0).astype(int)

#######################################

    # Calcola la mediana solo sul training
    #expendures_median = combined_df.loc[combined_df['IsTrain'], 'Expendures'].median()

    # Binarizza expendures
    #combined_df['Expendures'] = combined_df['Expendures'] > expendures_median

#######################################


    # Print dei conteggi per ogni categoria di AgeGroup
    #print("\nValori per ogni categoria di 'AgeGroup':")
    #print(combined_df['AgeGroup'].value_counts(dropna=False))



    #print("Valori mancanti nella colonna AgeGroup:", combined_df['AgeGroup'].isna().sum())


    # === 5. Ritaglia i dataset finali ===
    new_train = combined_df[combined_df['IsTrain']== True].copy()
    new_val   = combined_df[combined_df['IsValidation']== True].copy()
    new_test  = combined_df[combined_df['IsTest']== True].copy()

    # === 6. Salva i file finali ===
    new_train.to_excel('C:/Users/dvita/Desktop/TITANIC/train_df.xlsx', index=False)
    new_val.to_excel('C:/Users/dvita/Desktop/TITANIC/val_df.xlsx', index=False)
    new_test.to_excel('C:/Users/dvita/Desktop/TITANIC/test_df.xlsx', index=False)

    print("File salvati correttamente.")
    return combined_df

