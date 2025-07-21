import pandas as pd
import matplotlib.pyplot as plt

# === 1. Caricamento dataset (input dinamico) ===
csv_path = input("Inserisci il percorso del file train.csv: ").strip()
df = pd.read_csv(csv_path)

# === 2. Estrai 'Group' da 'PassengerId' ===
df['Group'] = df['PassengerId'].str.split('_').str[0]

# === 3. Rimuovi righe dove il valore 'VIP' è mancante (NaN) ===
df = df.dropna(subset=['VIP'])

# === 4. Calcola la dimensione dei gruppi ===
group_sizes = df['Group'].value_counts()

# === 5. Filtra il DataFrame per includere solo gruppi con almeno 2 persone ===
valid_groups = group_sizes[group_sizes >= 2].index
df = df[df['Group'].isin(valid_groups)]

# === 6. Raggruppa per Group e analizza ===
group_vip_status = df.groupby('Group')['VIP']
groups_with_at_least_one_vip = group_vip_status.any()
groups_all_vip = group_vip_status.all()

# === 7. Confronta: in quali gruppi c'è almeno un VIP ma non sono tutti VIP? ===
inconsistent_vip_groups = groups_with_at_least_one_vip & (~groups_all_vip)

# === 8. Output ===
total_groups = df['Group'].nunique()
num_inconsistent = inconsistent_vip_groups.sum()
num_consistent = total_groups - num_inconsistent
perc_inconsistent = (num_inconsistent / total_groups * 100)
perc_consistent = (num_consistent / total_groups * 100)

print("\nAnalisi della coerenza del campo 'VIP' nei gruppi con almeno 2 persone:")
print(f"Gruppi analizzati: {total_groups}")
print(f"Gruppi coerenti (tutti VIP oppure tutti NON VIP): {num_consistent} ({perc_consistent:.2f}%)")
print(f"Gruppi misti (solo alcuni membri VIP): {num_inconsistent} ({perc_inconsistent:.2f}%)")

# === 9. Grafico a barre ===
labels = ['Gruppi coerenti', 'Gruppi misti']
values = [num_consistent, num_inconsistent]
colors = ['skyblue', 'lightcoral']

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=colors)

# Etichette sopra le barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 1, str(height), ha='center', va='bottom')

plt.title("Coerenza del campo 'VIP' nei gruppi (≥2 membri)")
plt.ylabel("Numero di gruppi")
plt.tight_layout()
plt.show()
