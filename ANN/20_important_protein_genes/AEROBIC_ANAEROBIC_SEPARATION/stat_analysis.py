import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Read Excel file
df = pd.read_excel('20_proteins.xlsx')

# Get the list of proteins (columns from B to U)
proteins = df.columns[1:]

# Initialize variables to store significant and non-significant proteins
significant_proteins = []
nonsignificant_proteins = []

# Iterate over each protein
for protein in proteins:
    # Get data for _an and _ae conditions
    an_data = df[df['Conditions'].str.contains('_an')][protein]
    ae_data = df[df['Conditions'].str.contains('_ae')][protein]
   
    # Convert numpy arrays to pandas Series
    an_data = pd.Series(an_data)
    ae_data = pd.Series(ae_data)
   
    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(an_data, ae_data)
   
    # Adjust p-value for multiple comparisons
    adjusted_p_value = multipletests([p_value], method='fdr_bh')[1][0]
   
    # If p-value is less than or equal to 0.05, add protein to significant_proteins list
    if adjusted_p_value <= 0.05:
        significant_proteins.append((protein, adjusted_p_value))
    else:
        nonsignificant_proteins.append((protein, adjusted_p_value))

# Function to create subplots
def create_subplots(proteins, title):
    num_plots = len(proteins)
    num_cols = 2  # 2 columns for two graphs in one row
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Create subplots with size that fits a letter page (8.5 x 11 inches)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8.5, 11)) 

    # Flatten axes if necessary
    if num_rows == 1:
        axes = axes.flatten()
    else:
        axes = [ax for sublist in axes for ax in sublist]

    # Iterate over proteins and plot boxplots
    for i, (protein, adjusted_p_value) in enumerate(proteins):
        # Get data for _an and _ae conditions
        an_data = df[df['Conditions'].str.contains('_an')][protein]
        ae_data = df[df['Conditions'].str.contains('_ae')][protein]
       
        # Plot boxplots for _an and _ae conditions
        axes[i].boxplot([an_data, ae_data], patch_artist=True, widths=0.3,  # Reduce box width for a more compact fit
                        labels=['Anaerobic', 'Aerobic'],
                        boxprops=dict(facecolor='skyblue', edgecolor='black'), whiskerprops=dict(color='black'),
                        medianprops=dict(color='black'), capprops=dict(color='black'))

        # Set title, xlabel, ylabel for each subplot with bold, larger font sizes
        axes[i].set_title(f'{protein} (p adjusted: {adjusted_p_value:.4f})', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Abundance', fontsize=12, fontweight='bold')
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    # Remove unused subplots
    for j in range(len(proteins), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout, space between subplots, and add main title with larger, bold font
    fig.suptitle(title, fontsize=18, fontweight='bold')
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust top to make room for the title
    plt.show()

# Plot significant proteins
create_subplots(significant_proteins, "Differentially Abundant Proteins")

# Plot non-significant proteins
create_subplots(nonsignificant_proteins, "Non-Significantly Abundant Proteins")