import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv = pd.read_csv('dataset_final.csv') #apistox dataset

# chemicals that are uncategorized agrochemicals, not marked as herbicide, fungicide, insecticide or other_agrochemical
uncat_agrochem = csv[(csv['herbicide'] == 0) & (csv['fungicide'] == 0) & (csv['insecticide'] == 0) & (csv['other_agrochemical'] == 0)]
csv = csv.drop(uncat_agrochem.index)

#drop SMILES column, not needed for this analysis
csv = csv.drop(columns=['SMILES'])

# Distribution of compounds based on toxiciity class (ppdb_level)
plt.figure(figsize=(8,5))
ax = sns.countplot(data=csv, x='ppdb_level')
plt.title('Distribution of Compounds by PPDB Toxicity Level')
plt.xlabel('PPDB Toxicity Level')
plt.ylabel('Count')

# add percent labels to bars
total = len(csv)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()
# Quick analysis of resulting visualization shows a normal distribution with ppdb_level 1 being the most common

#4 subplots histograms side by side of herbicide vs. ppdb_level, fungicide vs. ppdb_level, insecticide vs. ppdb_level and other_agrochemical vs. ppdb_level
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(data=csv, discrete=True, x='ppdb_level', hue='herbicide', multiple='dodge', shrink= .33, ax=axs[0, 0])
axs[0, 0].set_title('Herbicide vs. PPDB Level')
sns.histplot(data=csv, discrete=True, x='ppdb_level', hue='fungicide', multiple='dodge', shrink= .33, ax=axs[0, 1])
axs[0,1].set_title('Fungicide vs. PPDB Level')
sns.histplot(data=csv, discrete=True, x='ppdb_level', hue='insecticide', multiple='dodge', shrink= .33, ax=axs[1, 0])
axs[1,0].set_title('Insecticide vs. PPDB Level')
sns.histplot(data=csv, discrete=True, x='ppdb_level', hue='other_agrochemical', multiple='dodge', shrink= .33, ax=axs[1, 1])
axs[1,1].set_title('Other Agrochemical vs. PPDB Level')
plt.show()

#subplots comparing toxicity_type vs ppdb_level for each toxicity_type (oral, contact, other)
sns.displot(data=csv, discrete=True, x='ppdb_level', hue='toxicity_type', multiple='dodge', shrink= .33)
plt.title('Toxicity Type vs. PPDB Level')
plt.xlabel('PPDB Toxicity Level')
plt.ylabel('Count')
plt.show()

# Distribution of compounds by year vs ppdb_level
#bin data into decades to reduce noise
csv['Decade'] = (csv['year'] // 10) * 10

# pivot table to count occurrences of each ppdb_level per decade
pivot_csv = csv.pivot_table(index='Decade', columns='ppdb_level', values='year', aggfunc='count').fillna(0)

# stacked par plot
plt.figure(figsize=(14,  8))
plt.stackplot(pivot_csv.index, pivot_csv.T, labels=pivot_csv.columns, colors=['yellow', 'orange', 'red'], alpha=0.8)
plt.title('Distribution of Compounds by Decade and PPDB Toxicity Level')
plt.xlabel('Decade')
plt.ylabel('Compound Count')
plt.legend(title='Toxicity Level')
plt.grid(True, alpha=0.3)
plt.show()