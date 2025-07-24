import pandas as pd
import numpy as np

def converti_valori_colonne(train_path, val_path, test_path,
                             output_train_path, output_val_path, output_test_path):
    train_df = pd.read_excel(train_path)
    val_df   = pd.read_excel(val_path)
    test_df  = pd.read_csv(test_path)

    # Flag identificativi
    train_df['IsTrain'] = True
    train_df['IsValidation'] = False
    train_df['IsTest'] = False

    val_df['IsTrain'] = False
    val_df['IsValidation'] = True
    val_df['IsTest'] = False

    test_df['IsTrain'] = False
    test_df['IsValidation'] = False
    test_df['IsTest'] = True

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Feature engineering
    combined_df['Group'] = combined_df['PassengerId'].str.split('_').str[0].astype(int)
    group_counts = combined_df['Group'].value_counts()
    combined_df['Group_size'] = combined_df['Group'].map(group_counts)

    combined_df[['Deck', 'CabinNum', 'Side']] = combined_df['Cabin'].str.split('/', expand=True)
    combined_df.drop(columns=['Cabin'], inplace=True)
    
    # Per convertire in numerico e lasciare NaN
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

    # Colonne di spesa
    spesa_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    combined_df[spesa_cols] = combined_df[spesa_cols].fillna(0)

    # Trasforma in log(1 + x)
    for col in spesa_cols:
        combined_df[f'Log_{col}'] = np.log1p(combined_df[col])

    # Droppa le colonne originali
    combined_df = combined_df.drop(columns=spesa_cols)

    # Somma delle spese
    combined_df['Expenditures'] = combined_df[[f'Log_{c}' for c in spesa_cols]].sum(axis=1)

    # Trasforma Transported in binario
    combined_df['Transported'] = combined_df['Transported'].map({True: 1, False: 0})

    # **Nuova parte: trasforma CryoSleep in binario (1=True, 0=False), preserva NaN**
    combined_df['CryoSleep'] = combined_df['CryoSleep'].map({True: 1, False: 0})

    # Ricava i DataFrame finali
    df_train = combined_df[combined_df['IsTrain'] == 1] \
        .drop(columns=['IsTrain', 'IsValidation', 'IsTest'])
    df_val   = combined_df[combined_df['IsValidation'] == 1] \
        .drop(columns=['IsTrain', 'IsValidation', 'IsTest'])
    df_test  = combined_df[combined_df['IsTest'] == 1] \
        .drop(columns=['IsTrain', 'IsValidation', 'IsTest'])

    # Salva
    df_train.to_excel(output_train_path, index=False)
    df_val.to_excel(output_val_path, index=False)
    df_test.to_excel(output_test_path, index=False)

    return combined_df
