import pandas as pd
import matplotlib.pyplot as plt


# === 1. Caricamento dataset (input dinamico) ===
csv_path = input("Inserisci il percorso del file train.csv: ").strip()
df = pd.read_csv(csv_path)

# === 2. Estrai il campo 'Group' dal PassengerId ===
df['Group'] = df['PassengerId'].str.split('_').str[0]

# === 3. Estrai il cognome dalla colonna Name ===
df = df.dropna(subset=['Name'])
df['Surname'] = df['Name'].str.split().str[-1]

# === 3.5 Conta quanti gruppi hanno una sola persona ===
group_sizes = df['Group'].value_counts()
single_member_groups = (group_sizes == 1).sum()
print(f"\nGruppi con una sola persona: {single_member_groups}")

# === 4. Conta quanti cognomi distinti ci sono per ogni gruppo ===
surname_counts_per_group = df.groupby('Group')['Surname'].nunique()

# === 5. Gruppi con un solo cognome ===
coherent_groups = (surname_counts_per_group == 1).sum()
total_groups = surname_counts_per_group.shape[0]
incoherent_groups = total_groups - coherent_groups
percentage = coherent_groups / total_groups * 100

# === 6. Output ===
print("\nAnalisi gruppi con unico cognome:")
print(f"Gruppi totali: {total_groups}")
print(f"Gruppi con un solo cognome: {coherent_groups}")
print(f"Percentuale: {percentage:.2f}%")

# === 7. Analisi dei gruppi non coerenti con più persone ===
multi_member_groups = group_sizes[group_sizes > 1].index
multi_member_surname_counts = surname_counts_per_group.loc[multi_member_groups]
non_coherent_multi_groups = multi_member_surname_counts[multi_member_surname_counts > 1].index

groups_with_prevalent_surname = 0

for group in non_coherent_multi_groups:
    surnames_in_group = df[df['Group'] == group]['Surname']
    surname_counts = surnames_in_group.value_counts()
    if len(surname_counts) > 1 and surname_counts.iloc[0] > surname_counts.iloc[1]:
        groups_with_prevalent_surname += 1

# === 8. Output analisi prevalenza ===
total_noncoherent_multi = len(non_coherent_multi_groups)
no_prevalent = total_noncoherent_multi - groups_with_prevalent_surname
print(f"\nGruppi con più persone e più cognomi (non coerenti): {total_noncoherent_multi}")
print(f"  - Di questi, gruppi con un cognome prevalente: {groups_with_prevalent_surname}")
if total_noncoherent_multi > 0:
    perc = (groups_with_prevalent_surname / total_noncoherent_multi) * 100
    print(f"  - Percentuale con cognome prevalente: {perc:.2f}%")

# === 9. Grafico 1: Coerenza gruppi (unico cognome vs multipli) ===
plt.figure(figsize=(7, 5))
labels1 = ['Coerenti', 'Non coerenti']
values1 = [coherent_groups, incoherent_groups]
colors1 = ['skyblue', 'lightcoral']
bars1 = plt.bar(labels1, values1, color=colors1)

for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 20, str(height), ha='center', va='bottom')

plt.title('Distribuzione gruppi per coerenza del cognome')
plt.ylabel('Numero di gruppi')
plt.tight_layout()
plt.show()

# === 10. Grafico 2: Cognome prevalente nei gruppi non coerenti ===
if total_noncoherent_multi > 0:
    plt.figure(figsize=(7, 5))
    labels2 = ['Prevalente', 'Nessun prevalente']
    values2 = [groups_with_prevalent_surname, no_prevalent]
    colors2 = ['mediumseagreen', 'lightgray']
    bars2 = plt.bar(labels2, values2, color=colors2)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5, str(height), ha='center', va='bottom')

    plt.title('Nei gruppi non coerenti: esiste un cognome prevalente?')
    plt.ylabel('Numero di gruppi non coerenti')
    plt.tight_layout()
    plt.show()

