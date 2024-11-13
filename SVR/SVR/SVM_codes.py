import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Load data from Excel file into pandas DataFrame
df = pd.read_excel('proteomics_data.xlsx', sheet_name='Sheet1')

# Extract features (proteomics data) and target (growth rate) from specific columns
X = df.iloc[:, 1:1855].values  # features
y = df['GR'].values  # target

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Define the SVR model
model = SVR(kernel='rbf', C=100, epsilon=0.01)

# Train the model
model.fit(X_scaled, y)

# Predict growth rate for training data
y_train = model.predict(X_scaled)

# Calculate mean absolute error and mean squared error for training data
mae = mean_absolute_error(y, y_train)
mse = mean_squared_error(y, y_train)
train_accuracy_percentage_no_rounding = (1 - np.mean(np.abs((y_train - y) / y))) * 100
print("Training Metrics:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Percentage accuracy:", train_accuracy_percentage_no_rounding)

# Plot actual vs predicted growth rate values
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.scatter(y, y_train, color='blue', label='Actual vs Predicted Growth Rate', alpha=0.7, edgecolors='w')
plt.plot(y, y, color='red', linestyle='-', label='Perfect Prediction Line', linewidth=2)  # Perfect prediction line (y = x)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Actual vs Predicted Growth Rate', fontsize=16)
plt.xlabel('Actual Growth Rate', fontsize=14)
plt.ylabel('Predicted Growth Rate', fontsize=14)
plt.legend(fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()