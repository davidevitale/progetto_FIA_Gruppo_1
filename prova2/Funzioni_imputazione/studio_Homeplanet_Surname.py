import pandas as pd
import matplotlib.pyplot as plt

# === 1. Caricamento dataset (input dinamico) ===
csv_path = input("Inserisci il percorso del file train.csv: ").strip()
df = pd.read_csv(csv_path)

# === 2. Rimuovi righe senza Name o HomePlanet ===
df = df.dropna(subset=['Name', 'HomePlanet'])

# === 3. Estrai il cognome ===
df['Surname'] = df['Name'].str.split().str[-1]

# === 4. Conta quante persone condividono ciascun cognome ===
surname_counts = df['Surname'].value_counts()

# === 5. Filtra solo i cognomi usati da almeno 2 persone distinte ===
shared_surnames = surname_counts[surname_counts > 1].index

# === 6. Per ciascun cognome condiviso, conta i valori distinti di HomePlanet ===
shared_df = df[df['Surname'].isin(shared_surnames)]
homeplanet_per_surname = shared_df.groupby('Surname')['HomePlanet'].nunique()

# === 7. Conta quanti cognomi condivisi sono coerenti (1 solo HomePlanet) ===
coherent = (homeplanet_per_surname == 1).sum()
incoherent = (homeplanet_per_surname > 1).sum()
total_shared = coherent + incoherent

# === 8. Output ===
print("\nAnalisi coerenza tra Surname e HomePlanet:")
print(f"Cognomi condivisi da più persone: {total_shared}")
print(f"  - Cognomi coerenti (tutti dallo stesso HomePlanet): {coherent}")
print(f"  - Cognomi incoerenti (HomePlanet diverso): {incoherent}")
if total_shared > 0:
    print(f"  - Percentuale coerenti: {coherent / total_shared * 100:.2f}%")

# === 9. Grafico a barre ===
if total_shared > 0:
    labels = ['Coerenti', 'Incoerenti']
    values = [coherent, incoherent]
    colors = ['skyblue', 'lightcoral']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)

    # Etichette numeriche sopra ogni barra
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, str(height), ha='center', va='bottom')

    plt.title('Cognomi condivisi: coerenza tra Surname e HomePlanet')
    plt.ylabel('Numero di cognomi')
    plt.tight_layout()
    plt.show()
