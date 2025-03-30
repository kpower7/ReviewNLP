import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

# Load the preprocessed DataFrame
df = pd.read_pickle('/home/gridsan/kpower/BERT_for_Amazon/processed_reviews.pkl')

# Sample for testing
# df = df.sample(n=1000, random_state=42)

# Combine 'tokens' and 'summarytokens' into a single text input
df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

# Load the pre-trained sentiment analysis model and tokenizer
sentiment_model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)

num_proc = min(32, cpu_count())  # Use up to 32 CPUs

def tokenize_texts(texts, tokenizer):
    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

# Split the data into chunks for parallel processing
chunk_size = len(df) // num_proc
text_chunks = [df['combined_text'].tolist()[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

# Parallel tokenization
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    results = list(executor.map(lambda x: tokenize_texts(x, sentiment_tokenizer), text_chunks))

# Merge tokenized results
sentiment_inputs = {
    'input_ids': torch.cat([r['input_ids'] for r in results], dim=0),
    'attention_mask': torch.cat([r['attention_mask'] for r in results], dim=0)
}

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
    3: "Incorrect",
    4: "Quality Issue"
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
