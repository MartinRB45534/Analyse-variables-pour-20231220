import pandas as pd
from sklearn import metrics as skm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Specify the savefile path (relative to the current working directory)
savefile_path = "data/211117_Batiment_GG.csv"

# Try to load the csv file
try:
    df = pd.read_csv(savefile_path)

except:
    # Specify the file path
    file_path = "data/211117_Batiment_GG.xlsx"

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Save the file
    df.to_csv(savefile_path, index=False)

# Remove unnecessary columns
df_clean = df.drop(['OBJECTID', 'EGID', 'LIEU', 'COMMUNE'], axis=1)
column_names = df_clean.columns.tolist()

# Print the column names
print("Column names:")
print(column_names)

# Drop the rows with missing values
len_before = len(df_clean)
df_clean = df_clean.dropna()
len_after = len(df_clean)
print(f"Number of rows dropped: {len_before - len_after} ({(len_before - len_after) / len_before * 100}%)")
print(f"Remaining na values: {df_clean.isna().sum().sum()}") # Should be 0

# Calculate the mutual information matrix
mutual_info_matrix = np.zeros((len(column_names), len(column_names)))
for i in range(len(column_names)):
    for j in range(len(column_names)):
        print(f"Coefficient for {column_names[i]} and {column_names[j]}")
        mutual_info_matrix[i, j] = skm.mutual_info_score(df_clean[column_names[i]], df_clean[column_names[j]])

# Print the mutual information matrix
print("Mutual Information Matrix:")
print(mutual_info_matrix)

# Show the mutual information matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(mutual_info_matrix, annot=True, fmt='.2f', xticklabels=column_names, yticklabels=column_names, cmap='YlGnBu')
# xlabels above the heatmap
plt.gca().xaxis.tick_top()
# Add title and legend and everything
plt.title("Mutual Information Matrix")
plt.xlabel("Column Names")
plt.ylabel("Column Names")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Calculate the correlation matrix
correlation_matrix = df_clean.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Show the correlation matrix as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', xticklabels=column_names, yticklabels=column_names, cmap='YlGnBu')
# xlabels above the heatmap
plt.gca().xaxis.tick_top()
# Add title and legend and everything
plt.title("Correlation Matrix")
plt.xlabel("Column Names")
plt.ylabel("Column Names")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Shannon entropy
shannon_entropy = np.zeros(len(column_names))
for i in range(len(column_names)):
    shannon_entropy[i] = stats.entropy(df_clean[column_names[i]].value_counts(normalize=True))

# Print the Shannon entropy
print("Shannon Entropy:")
print(shannon_entropy)

# Show the Shannon entropy as a bar plot
plt.figure(figsize=(10, 10))
plt.bar(column_names, shannon_entropy)
# Add title and legend and everything
plt.title("Shannon Entropy")
plt.xlabel("Column Names")
plt.ylabel("Shannon Entropy")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot
plt.show()

