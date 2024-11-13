import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# Set random seed for reproducibility
np.random.seed(42)

# Load data from Excel file into pandas DataFrame
df = pd.read_excel('proteomics_data.xlsx', sheet_name='Sheet1')

# Extract features (proteomics data) and target (growth rate) from specific columns
X = df.iloc[:, 1:1855].values  # features
y = df['GR'].values  # target

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X)

# Define the MLP regressor model
model = MLPRegressor(hidden_layer_sizes=(200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
                     max_iter=1000,
                     random_state=42)

# Train the model
model.fit(X_train_scaled, y)

# Predict growth rate for training data
y_train = model.predict(X_train_scaled)

# Calculate residuals
residuals = y_train - model.predict(X_train_scaled)

# Calculate regression line and Pearson correlation coefficient
regression_line = np.polyfit(y_train, model.predict(X_train_scaled), 1)
slope, intercept = regression_line
correlation_coefficient, _ = pearsonr(y_train, model.predict(X_train_scaled))

# Plot actual vs predicted growth rate values
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.scatter(y, model.predict(X_train_scaled), color='blue', label='Actual vs Predicted Growth Rate', alpha=0.7, edgecolors='w')
plt.plot(y, y, color='red', linestyle='-', label='Perfect Prediction Line', linewidth=2)  # Perfect prediction line (y = x)
plt.xticks(fontsize=12, weight='bold')  # Bold x-ticks
plt.yticks(fontsize=12, weight='bold')  # Bold y-ticks
plt.title('Actual vs Predicted Growth Rate', fontsize=16, weight='bold')  # Bold title
plt.xlabel('Actual Growth Rate', fontsize=14, weight='bold')  # Bold x-axis label
plt.ylabel('Predicted Growth Rate', fontsize=14, weight='bold')  # Bold y-axis label
plt.legend(fontsize=12)  # Removed weight='bold' from here
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Initialize a figure and set its size
plt.figure(figsize=(12, 8))

# Loop through each sample
for i in range(len(residuals)):
    # Create a subplot for each sample
    plt.subplot(5, 5, i+1)
    
    # Plot the histogram of residuals for the current sample
    plt.hist(residuals[i], bins=20, color='skyblue', edgecolor='black')
    
    # Set title and labels
    plt.title(f'Residuals - Sample {i+1}', fontsize=12, weight='bold')
    plt.xlabel('Residuals', fontsize=10, weight='bold')
    plt.ylabel('Frequency', fontsize=10, weight='bold')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# Assuming y contains actual growth rate values and model.predict(X_train_scaled) contains predicted growth rate values
actual_vs_predicted = pd.DataFrame({'Actual Growth Rate': y, 'Predicted Growth Rate': model.predict(X_train_scaled)})

# Exporting to Excel
actual_vs_predicted.to_excel('actual_vs_predicted_growth_rate.xlsx', index=False)

# Calculate permutation feature importance with adjusted max_evals
perm_importance = permutation_importance(model, X_train_scaled, y_train, n_repeats=30, random_state=42)

# Get permutation importance for all features
all_importances = perm_importance.importances_mean

# Create a DataFrame to store feature names and their importance scores
feature_names = df.columns[1:1855]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': all_importances})

# Sort the DataFrame by importance scores in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Export the DataFrame to an Excel file
importance_df.to_excel('feature_importance_all_features.xlsx', index=False)

# Get indices of most important features based on permutation importance
top_indices_perm = np.abs(perm_importance.importances_mean).argsort()[-20:][::-1]

# Extract names of most important proteins based on permutation importance
top_proteins_perm = df.columns[1:1855][top_indices_perm]

# Reorder columns of abundance_data based on importance scores for permutation feature importance
abundance_data_perm = df[top_proteins_perm]

# Plot permutation feature importance
plt.figure(figsize=(10, 6))
plt.barh(top_proteins_perm, perm_importance.importances_mean[top_indices_perm], color='skyblue', edgecolor='black')
plt.xlabel('Permutation Importance', fontsize=12, weight='bold')
plt.title('Permutation Feature Importance', fontsize=14, weight='bold')
plt.grid(True)
plt.tight_layout()
plt.grid(False)
plt.show()

# Create box plot for most important proteins based on permutation importance
plt.figure(figsize=(10, 6))
sns.boxplot(data=abundance_data_perm, palette='Set2')
plt.title('Top 20 Most Important Proteins', fontsize=14, weight='bold')
plt.xlabel('Proteins', fontsize=12, weight='bold')
plt.ylabel('Protein Level', fontsize=12, weight='bold')
plt.xticks(rotation=45, fontsize=10, weight='bold')
plt.tight_layout()
plt.grid(False)
plt.show()

# Create clustermap for most important proteins based on permutation importance
plt.figure(figsize=(8, 6))
cluster_map = sns.clustermap(abundance_data_perm, cmap='viridis', method='average', standard_scale=1, linewidths=1, linecolor='black')
cluster_map.ax_heatmap.set_yticklabels(df.iloc[:, 0], rotation=0, fontsize=10, weight='bold')  # Assuming the actual conditions are in the first column
plt.title('Clustermap for Top 20 Most Important Proteins', fontsize=14, weight='bold')
plt.xlabel('Proteins', fontsize=12, weight='bold')
plt.ylabel('Conditions', fontsize=12, weight='bold')  # Change y-axis label
plt.tight_layout()
plt.grid(False)
plt.show()

# Calculate evaluation metrics
mse = mean_squared_error(y_train, y)
mae = mean_absolute_error(y_train, y)

# Calculate percentage error
train_accuracy_percentage_no_rounding = (1 - np.mean(np.abs((y_train - y) / y))) * 100

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Training Accuracy Percentage: {train_accuracy_percentage_no_rounding}%')