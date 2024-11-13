import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the Excel file
df = pd.read_excel('biomass_reduction.xlsx')

# Setting the positions for side-by-side bars
bar_width = 0.35
index = np.arange(len(df))

# Setting up a nice color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Comparison of Growth Rates Before and After Knockout using side-by-side bar graphs
plt.figure(figsize=(10, 6))
bars1 = plt.bar(index - bar_width/2, df['Actual'], bar_width, color=colors[0], edgecolor='black', label='Actual Growth Rate', alpha=0.8)
bars2 = plt.bar(index + bar_width/2, df['Knockout'], bar_width, color=colors[1], edgecolor='black', alpha=0.8)
plt.xlabel('Proteins', fontsize=12)
plt.ylabel('Growth Rate (/day)', fontsize=12)
plt.title('Comparison of Growth Rates Before and After Knockout', fontsize=14)
plt.xticks(index, df['Proteins'], rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Custom legend inside bars
legend_labels = ['Actual Growth Rate (/day)', 'Growth Rate After Knockout (/day)']
for i, label in enumerate(legend_labels):
    plt.text(0, 0, label, color=colors[i], bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

# Placing legend in the top left corner
plt.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

# Percentage of Reduction of Biomass for Each Protein After the Knockout
plt.figure(figsize=(10, 6))
plt.bar(df.index, df['Percentage_reduction'], color=colors[2], edgecolor='black', alpha=0.8)
plt.xlabel('Proteins', fontsize=12)
plt.ylabel('Growth Rate Reduction Percentage', fontsize=12)
plt.title('Percentage of Growth Rate Reduction for Each Protein After the Knockout', fontsize=14)
plt.xticks(df.index, df['Proteins'], rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()