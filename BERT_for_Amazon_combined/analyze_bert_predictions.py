import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('C:\\Users\\k_pow\\OneDrive\\Documents\\MIT\\MITx SCM\\Spring 2025\\SCM256\\Amazon_Local_Transfer\\BERT_for_Amazon_combined\\predicted_analysis_with_cause_expanded.csv')

# Calculate accuracy for 5-bin ratings
accuracy_5bin = accuracy_score(df['overall'], df['predicted'])
print("\n=== 5-Bin Rating Analysis ===")
print(f"Accuracy with 5-bin ratings: {accuracy_5bin:.4f}")

# Display confusion matrix for 5-bin
conf_matrix_5bin = confusion_matrix(df['overall'], df['predicted'])
print("\nConfusion Matrix (5-bin):")
print(conf_matrix_5bin)

# Function to convert 5-bin to 3-bin
def convert_to_3bin(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# Convert both columns to 3-bin ratings
df['actual_3bin'] = df['overall'].apply(convert_to_3bin)
df['predicted_3bin'] = df['predicted'].apply(convert_to_3bin)

# Calculate accuracy for 3-bin ratings
accuracy_3bin = accuracy_score(df['actual_3bin'], df['predicted_3bin'])
print("\n=== 3-Bin Rating Analysis ===")
print(f"Accuracy with 3-bin ratings: {accuracy_3bin:.4f}")

# Display confusion matrix for 3-bin
conf_matrix_3bin = confusion_matrix(df['actual_3bin'], df['predicted_3bin'])
print("\nConfusion Matrix (3-bin):")
print(conf_matrix_3bin)

# Additional statistics
print("\n=== Distribution of Ratings ===")
print("\nOriginal 5-bin ratings distribution:")
print("Actual ratings:", df['overall'].value_counts().sort_index())
print("Predicted ratings:", df['predicted'].value_counts().sort_index())

print("\n3-bin ratings distribution:")
print("Actual ratings:", df['actual_3bin'].value_counts())
print("Predicted ratings:", df['predicted_3bin'].value_counts())

# Create visualizations
plt.figure(figsize=(15, 5))

# Plot confusion matrix for 5-bin ratings
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_5bin, annot=True, fmt='d', cmap='Blues')
plt.title('5-Bin Ratings Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot confusion matrix for 3-bin ratings
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_3bin, annot=True, fmt='d', cmap='Blues')
plt.title('3-Bin Ratings Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Calculate and display additional metrics for 3-bin classification
categories = ['negative', 'neutral', 'positive']
conf_matrix_3bin_df = pd.DataFrame(conf_matrix_3bin, 
                                 index=categories, 
                                 columns=categories)

print("\n=== Detailed 3-Bin Classification Metrics ===")
for category in categories:
    TP = conf_matrix_3bin_df.loc[category, category]
    FP = conf_matrix_3bin_df[category].sum() - TP
    FN = conf_matrix_3bin_df.loc[category].sum() - TP
    TN = conf_matrix_3bin_df.values.sum() - (TP + FP + FN)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{category.capitalize()} Class Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
