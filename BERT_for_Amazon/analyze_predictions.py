import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('test_predictions.csv')

# Calculate accuracy for 5-bin ratings
accuracy_5bin = accuracy_score(df['actual'], df['predicted'])
print("\n=== 5-Bin Rating Analysis ===")
print(f"Accuracy with 5-bin ratings: {accuracy_5bin:.4f}")

# Display confusion matrix for 5-bin
conf_matrix_5bin = confusion_matrix(df['actual'], df['predicted'])
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
df['actual_3bin'] = df['actual'].apply(convert_to_3bin)
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
print("Actual ratings:", df['actual'].value_counts().sort_index())
print("Predicted ratings:", df['predicted'].value_counts().sort_index())

print("\n3-bin ratings distribution:")
print("Actual ratings:", df['actual_3bin'].value_counts())
print("Predicted ratings:", df['predicted_3bin'].value_counts())
