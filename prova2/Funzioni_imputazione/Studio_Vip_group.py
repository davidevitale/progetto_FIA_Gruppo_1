import pandas as pd
import matplotlib.pyplot as plt

# === 1. Caricamento dataset (input dinamico) ===
csv_path = input("Inserisci il percorso del file train.csv: ").strip()
df = pd.read_csv(csv_path)

# === 2. Estrai 'Group' da 'PassengerId' ===
df['Group'] = df['PassengerId'].str.split('_').str[0]

# === 3. Rimuovi righe dove il valore 'VIP' è mancante (NaN) ===
df = df.dropna(subset=['VIP'])

# === 4. Raggruppa per Group e analizza ===
group_vip_status = df.groupby('Group')['VIP']

groups_with_at_least_one_vip = group_vip_status.any()
groups_all_vip = group_vip_status.all()

# === 5. Confronta: in quali gruppi c'è almeno un VIP ma non sono tutti VIP? ===
inconsistent_vip_groups = groups_with_at_least_one_vip & (~groups_all_vip)

# === 6. Output ===
total_groups = df['Group'].nunique()
num_inconsistent = inconsistent_vip_groups.sum()
num_consistent = total_groups - num_inconsistent
perc_inconsistent = (num_inconsistent / total_groups * 100)
perc_consistent = (num_consistent / total_groups * 100)

print("\nAnalisi della coerenza del campo 'VIP' all'interno dei gruppi:")
print(f"Gruppi analizzati (con valori disponibili nel campo VIP): {total_groups}")
print(f"Gruppi coerenti (tutti VIP oppure tutti NON VIP): {num_consistent} ({perc_consistent:.2f}%)")
print(f"Gruppi in cui solo alcuni membri sono VIP (misti): {num_inconsistent}")
print(f"Percentuale di gruppi misti (VIP/non VIP): {perc_inconsistent:.2f}%")


# === 7. Grafico a barre ===
labels = ['Gruppi coerenti', 'Gruppi misti']
values = [num_consistent, num_inconsistent]
colors = ['skyblue', 'lightcoral']

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=colors)

# Etichette sopra le barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 5, str(height), ha='center', va='bottom')

plt.title("Coerenza del campo 'VIP' nei gruppi")
plt.ylabel("Numero di gruppi")
plt.tight_layout()
plt.show()
