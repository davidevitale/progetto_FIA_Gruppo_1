# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:51:57 2025

@author: matmi
"""
import pandas as pd

# === 1. Caricamento del dataset (input dinamico) ===
csv_path = input("Inserisci il percorso del file train.csv: ").strip()
df = pd.read_csv(csv_path)

# === 2. Estrai Group dal PassengerId ===
df['Group'] = df['PassengerId'].str.split('_').str[0]

# === 3. Estrai il cognome ===
df['Surname'] = df['Name'].str.split().str[-1]

# === 4. Conta i cognomi distinti per gruppo ===
group_surname_counts = df.groupby('Group')['Surname'].nunique()

# === 5. Gruppi coerenti (1 solo cognome) ===
coherent_groups = (group_surname_counts == 1).sum()
total_groups = group_surname_counts.shape[0]
coherent_percentage = coherent_groups / total_groups * 100

# === 6. Righe con Surname mancante ===
missing_surnames = df['Surname'].isna().sum()

# === 7. Output dei risultati ===
print("\nVerifica se Group → Surname è affidabile")
print(f"Gruppi totali: {total_groups}")
print(f"Gruppi con 1 solo cognome: {coherent_groups}")
print(f"Percentuale di gruppi coerenti: {coherent_percentage:.2f}%")
print(f"Righe con Surname mancante: {missing_surnames}")


