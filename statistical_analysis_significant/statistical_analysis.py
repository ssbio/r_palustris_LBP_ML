import pandas as pd
import matplotlib.pyplot as plt

# Read data from the Excel file
file_path = 'proteins.xlsx'  # Assuming the Excel file is in the same directory as the Python script
df = pd.read_excel(file_path)

# Identify the condition columns
conditions = df.columns[1:-3]  # Exclude 'Conditions', 'Standard_Deviation', 'Mean', 'CV'

# Calculate standard deviation and mean for each protein across conditions
df['Standard_Deviation'] = df[conditions].std(axis=1)
df['Mean'] = df[conditions].mean(axis=1)
df['CV'] = df['Standard_Deviation'] / df['Mean']

# Sort by CV
df_sorted = df.sort_values(by='CV')

# Plot the bar chart
plt.figure(figsize=(12, 8))
plt.barh(df_sorted['Conditions'], df_sorted['CV'], color='skyblue', edgecolor='black')
plt.xlabel('Coefficient of Variation (CV)')
plt.title('Coefficient of Variation for Different Genes')

# Save and show the plot
plt.tight_layout()
plt.savefig('genes_cv_bar_chart.png')
plt.show()