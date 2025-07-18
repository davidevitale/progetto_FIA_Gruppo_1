import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#df = pd.read_excel('C:/Users/Standard/Desktop/Titanic/Titanic/train_holdout.xlsx')

# Ora carichi uno dei file di output per lavorare con il DataFrame
df = pd.read_excel('C:/Users/dvita/Desktop/TITANIC/train.xlsx')

# Nuova feature - Group (presupponendo che 'PassengerId' sia nel formato '123_45')
df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)

# Nuova feature - Group size
group_counts = df['Group'].value_counts()
df['Group_size'] = df['Group'].map(group_counts)


# Estrae il cognome dalla colonna Name
df['Surname'] = df['Name'].str.split().str[-1]

palette = {True: 'green', False: 'red'}

# Pie plot con colori personalizzati
df['Transported'].value_counts().plot.pie(
    explode=[0.1, 0.1],
    autopct='%1.1f%%',
    shadow=True,
    textprops={'fontsize': 16},
    colors=[palette[False], palette[True]]
).set_title("Target distribution")

plt.ylabel('')  # Rimuove etichetta dell'asse y
plt.show()

# Figure size
plt.figure(figsize=(20,4))

# Histogram Group
sns.histplot(data=df, x='Group', hue='Transported', binwidth=1, kde=True, palette=palette)
plt.title('Group')
plt.xlabel('Group')
plt.ylabel('Numero di passeggeri')

plt.subplot(1,2,2)
sns.countplot(data=df, x='Group_size', hue='Transported', palette=palette)
plt.title('Group size')
plt.tight_layout()

# New feature
df['Solo']=(df['Group_size']==1).astype(int)

# New feature distribution
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='Solo', hue='Transported', palette=palette)
plt.title('Passenger travelling solo or not')
plt.ylim([0,3000])


# Estrae le componenti della Cabina
df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='CabinNum', hue='Transported', binwidth=20, palette=palette)
plt.vlines(300, ymin=0, ymax=200, color='black')
plt.vlines(600, ymin=0, ymax=200, color='black')
plt.vlines(900, ymin=0, ymax=200, color='black')
plt.vlines(1200, ymin=0, ymax=200, color='black')
plt.vlines(1500, ymin=0, ymax=200, color='black')
plt.vlines(1800, ymin=0, ymax=200, color='black')
plt.title('Cabin number')
plt.xlim([0, 2000])
plt.show()

# New features - training set
df['Cabin_region1'] = (df['CabinNum'] < 300).astype(int)  # one-hot encoding
df['Cabin_region2'] = ((df['CabinNum'] >= 300) & (df['CabinNum'] < 600)).astype(int)
df['Cabin_region3'] = ((df['CabinNum'] >= 600) & (df['CabinNum'] < 900)).astype(int)
df['Cabin_region4'] = ((df['CabinNum'] >= 900) & (df['CabinNum'] < 1200)).astype(int)
df['Cabin_region5'] = ((df['CabinNum'] >= 1200) & (df['CabinNum'] < 1500)).astype(int)
df['Cabin_region6'] = ((df['CabinNum'] >= 1500) & (df['CabinNum'] < 1800)).astype(int)
df['Cabin_region7'] = (df['CabinNum'] >= 1800).astype(int)


df['AgeGroup'] = df['Age'].apply(
    lambda age: np.nan if pd.isna(age) else
                '0-18' if age <= 18 else
                '19-25' if age <= 25 else
                '25+'
)



exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

for i, feat in enumerate(exp_feats):
    sns.histplot(data=df, x=feat, bins=30, kde=False, hue='Transported', palette=palette, ax=axes[i])
    # Rimuovo il titolo
    # axes[i].set_title(f'Distribuzione spese per {feat}')
    axes[i].set_xlabel(feat, labelpad=15)  # alza l’etichetta x
    axes[i].set_ylabel('Numero di passeggeri')

# Elimina eventuali subplot vuoti
if len(exp_feats) < len(axes):
    for j in range(len(exp_feats), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # aumenta lo spazio verticale tra i plot
plt.show()


df['Expenditures'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) #skipna=True)

# Calcolo della mediana
expenditures_median = df['Expenditures'].median()
print(expenditures_median)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Expendures', bins=30, kde=False, hue='Transported', palette=palette)
plt.title('Distribuzione delle spese totali (Expenditures)')
plt.xlabel('Expenditures')
plt.ylabel('Numero di passeggeri')
plt.show()

# Filtro per Expendures < mediana
filtered_data = df[df['Expenditures'] <= expenditures_median]
x2 = df[df['Expenditures'] > expenditures_median]

# Conta quanti sono Transported = True e False
transported_counts1 = filtered_data['Transported'].value_counts()
transported_counts2 = x2['Transported'].value_counts()

# Stampa i conteggi
print("Conto di 'Transported' per passeggeri con Expendures < mediana:")
print(transported_counts1)
print("Conto di 'Transported' per passeggeri con Expendures > mediana:")
print(transported_counts2)

# Filtro i passeggeri con CryoSleep = False
cryo_false = df[df['CryoSleep'] == False]

# Calcola la mediana delle spese
mediana_expenditures = cryo_false['Expenditures'].median()

# Stampa il risultato
print(f"Mediana delle Expendures per passeggeri con CryoSleep = False: {mediana_expenditures}")

# Creazione della feature binaria
df['Expenditures'] = (df['Expendures'] > expenditures_median)

cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Expendures']

# Prima figura con i primi 3 grafici
plt.figure(figsize=(10, 12))
for i, var_name in enumerate(cat_feats[:3]):
    ax = plt.subplot(3,1,i+1)
    sns.countplot(data=df, x=var_name, hue='Transported', palette=palette, ax=ax)
    ax.xaxis.labelpad = -5
    if i < 2:  # rimuove la legenda da tutti tranne l'ultimo
        ax.get_legend().remove()
plt.tight_layout()
plt.show()

# Seconda figura con gli ultimi 3 grafici
plt.figure(figsize=(10, 12))
for i, var_name in enumerate(cat_feats[3:]):
    ax = plt.subplot(3,1,i+1)
    sns.countplot(data=df, x=var_name, hue='Transported', palette=palette, ax=ax)
    ax.xaxis.labelpad = -5
    # Mostra legenda solo nel primo plot della seconda figura
    if i == 0:
        ax.legend(title='Transported')
    else:
        ax.get_legend().remove()
plt.tight_layout()
plt.show()


# Figure size
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=df, x='Age', hue='Transported', binwidth=1, kde=True, palette=palette)

# Aesthetics
plt.title('Age distribution')
plt.xlabel('Age (years)')
plt.show()

# # 1. Crea la colonna AgeGroup (senza funzione esterna)
# df['AgeGroup'] = df['Age'].apply(
#     lambda age: 'Unknown' if pd.isna(age) else
#                 '0-18' if age <= 18 else
#                 '19-25' if age <= 25 else
#                 '25+'
# )

# # 2. Calcola le percentuali di 'Transported' per fascia d'età
# age_group_stats = data.groupby('AgeGroup')['Transported'].value_counts(normalize=True).unstack().fillna(0) * 100
# age_group_stats = age_group_stats.round(2)

# # 3. Mostra la tabella in console
# print("Percentuali di passeggeri 'Transported' per fascia d'età:\n")
# print(age_group_stats)

# # 4. Grafico a barre delle percentuali di 'True'
# plt.figure(figsize=(8, 5))
# age_group_stats[True].plot(kind='bar', color='lightblue')
# plt.title("Percentuale di 'Transported = True' per fascia d'età")
# plt.ylabel("Percentuale (%)")
# plt.xlabel("Fascia d'età")
# plt.xticks(rotation=0)
# plt.ylim(0, 100)
# plt.tight_layout()
# plt.show()

# Calcola il numero di HomePlanet unici per gruppo
df_group_hp = df.groupby('Group')['HomePlanet'].nunique().reset_index(name='UniqueHomePlanets')

# Crea una nuova colonna testuale per plotting
df_group_hp['PlanetCount'] = df_group_hp['UniqueHomePlanets'].astype(str)

# Plot: countplot per il numero di pianeti unici nei gruppi
plt.figure(figsize=(10, 4))
ax = plt.subplot(1, 1, 1)
sns.countplot(data=df_group_hp, x='PlanetCount', ax=ax)

# Estetica
ax.set_xlabel('Numero di pianeti unici nel gruppo')
ax.set_ylabel('Numero di gruppi')
ax.set_title('Distribuzione del numero di HomePlanet per gruppo')
plt.tight_layout()
plt.show()

# Distribuzione deck e home planet
deck_hp=df.groupby(['Deck','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

# Heatmap di missing values
plt.figure(figsize=(10,4))
sns.heatmap(deck_hp.T, annot=True, fmt='g', cmap='coolwarm')
plt.figure(figsize=(6,6))

# Joint distribution of HomePlanet and Destination
hp_dest=df.groupby(['HomePlanet','Destination'])['Destination'].size().unstack().fillna(0)

# Heatmap of missing values
plt.figure(figsize=(10,4))
sns.heatmap(hp_dest.T, annot=True, fmt='g', cmap='coolwarm')


# 1. Conta quante volte ogni cognome appare (escludendo NaN)
surname_counts = df['Surname'].value_counts()

# 2. Crea una colonna booleana: 1 se unico, 0 se condiviso, NaN se surname mancante
df['UniqueSurname'] = df['Surname'].map(lambda x: 1 if pd.notna(x) and surname_counts[x] == 1 else 0)

# 3. Grafico a barre con hue=Transported
plt.figure(figsize=(6, 5))
palette = palette  # Verde per True, rosso per False
sns.countplot(data=df, x='UniqueSurname', hue='Transported', palette=palette)

# 4. Etichette e layout
plt.title('Distribuzione di Transported rispetto a unicità del cognome')
plt.xlabel('Cognome Unico (1 = sì, 0 = no)')
plt.ylabel('Conteggio')
plt.xticks([0, 1], ['Condiviso', 'Unico'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Calcola dimensione dei gruppi
group_sizes = df['Group'].value_counts()
gruppi_validi = group_sizes[group_sizes >= 2].index  # gruppi con almeno 2 persone

# 3. Filtra solo gruppi validi e righe con VIP non nullo
df_valid = df[df['Group'].isin(gruppi_validi) & df['VIP'].notna()].copy()

# 4. Funzione: calcola la percentuale del valore VIP più frequente nel gruppo
def percentuale_vip_uguali(gruppo):
    mode = gruppo['VIP'].mode()
    if mode.empty:
        return None
    return (gruppo['VIP'] == mode.iloc[0]).mean()

# 5. Applica funzione a ciascun gruppo valido
percentuali = df_valid.groupby('Group').apply(percentuale_vip_uguali)

# 6. Calcola la media delle percentuali
media_percentuali_vip = round(percentuali.mean() * 100, 2)

# 7. Stampa il risultato
print(f"Percentuale media di VIP coerenti all'interno dei gruppi: {media_percentuali_vip}%")

# Filtra tutti i passeggeri con CryoSleep = True
cryosleep_true = df[df['CryoSleep'] == True]

# Calcola quanti hanno Expendures == 0 all'interno di quelli in CryoSleep
num_zero_expendures = (cryosleep_true['Expendures'] == 0).sum()

# Totale passeggeri in CryoSleep
total_cryosleep = len(cryosleep_true)

# Percentuale
percentuale = num_zero_expendures / total_cryosleep * 100

# Stampa il risultato
print(f"Percentuale di passeggeri con Expendures = 0 tra quelli con CryoSleep = True: {percentuale:.2f}%")
