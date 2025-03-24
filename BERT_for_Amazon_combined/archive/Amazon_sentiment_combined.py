import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# Load the preprocessed DataFrame
df = pd.read_pickle('/home/gridsan/kpower/BERT_for_Amazon/processed_reviews.pkl')

# Combine 'tokens' and 'summarytokens' into a single text input
df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

# Load the pre-trained sentiment analysis model and tokenizer
sentiment_model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)

# Tokenize the test set for sentiment analysis
sentiment_inputs = sentiment_tokenizer(list(df['combined_text']), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Create a DataLoader for batch processing
dataset = TensorDataset(sentiment_inputs['input_ids'], sentiment_inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=32)  # Adjust batch size as needed

# Perform sentiment analysis
sentiment_model.eval()
all_predictions = []

with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask = batch
        outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1) + 1  # Adjust to range 1-5
        all_predictions.extend(predictions.numpy())

# Add sentiment predictions to the dataframe
df['predicted'] = all_predictions

# Calculate accuracy and F1 score
accuracy = accuracy_score(df['overall'], df['predicted'])
f1 = f1_score(df['overall'], df['predicted'], average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# Load the LowRating fine-tuned model and tokenizer
low_rating_model_path = '/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model'
low_rating_tokenizer = BertTokenizer.from_pretrained(low_rating_model_path)
low_rating_model = BertForSequenceClassification.from_pretrained(low_rating_model_path)

# Filter rows with predicted rating of 1 or 2
low_rating_df = df[df['predicted'].isin([1, 2])]

# Tokenize the low rating rows for cause classification
low_rating_inputs = low_rating_tokenizer(list(low_rating_df['combined_text']), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Define a mapping from numeric cause predictions to category names
cause_mapping = {
    0: "Damaged",
    1: "Expired",
    2: "Delivery Issue",
    3: "Quality Issue",
    4: "Incorrect"
}

# Perform cause classification
low_rating_model.eval()
with torch.no_grad():
    low_rating_outputs = low_rating_model(**low_rating_inputs)
    low_rating_causes = torch.argmax(low_rating_outputs.logits, dim=1)

# Map numeric cause predictions to category names
low_rating_df['cause'] = low_rating_causes.numpy()
low_rating_df['cause'] = low_rating_df['cause'].map(cause_mapping)

# Merge the cause predictions back into the original dataframe
df.loc[low_rating_df.index, 'cause'] = low_rating_df['cause']

# Save the dataframe with sentiment and cause analysis
output_path = '/home/gridsan/kpower/BERT_for_Amazon_combined/predicted_analysis_with_cause.csv'
df.to_csv(output_path, index=False)

print("Test predictions with causes saved to CSV.")
