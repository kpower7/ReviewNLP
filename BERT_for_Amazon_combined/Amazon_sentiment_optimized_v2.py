import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

# Load Data
df = pd.read_pickle('/home/gridsan/kpower/BERT_for_Amazon/processed_reviews.pkl')

#sample for testing
# df = df.sample(n=10000, random_state=42)

# Combine 'tokens' and 'summarytokens' into a single text input
df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

# Load Model & Tokenizer
sentiment_model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)

# Move Model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

# Parallel Tokenization
num_proc = min(32, cpu_count())

def tokenize_texts(texts, tokenizer):
    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

text_chunks = [df['combined_text'].tolist()[i:i + len(df) // num_proc] for i in range(0, len(df), len(df) // num_proc)]

sentiment_inputs = {'input_ids': [], 'attention_mask': []}

with ThreadPoolExecutor(max_workers=num_proc) as executor:
    for batch in executor.map(lambda x: tokenize_texts(x, sentiment_tokenizer), text_chunks):
        sentiment_inputs['input_ids'].append(batch['input_ids'])
        sentiment_inputs['attention_mask'].append(batch['attention_mask'])

sentiment_inputs = {
    'input_ids': torch.cat(sentiment_inputs['input_ids'], dim=0).to(device),
    'attention_mask': torch.cat(sentiment_inputs['attention_mask'], dim=0).to(device)
}

# DataLoader for Batched Inference
dataset = TensorDataset(sentiment_inputs['input_ids'], sentiment_inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=32)

all_predictions = []

sentiment_model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1) + 1
        all_predictions.extend(predictions.cpu().tolist())

df['predicted'] = all_predictions

# Accuracy & F1 Score
accuracy = accuracy_score(df['overall'], df['predicted'])
f1 = f1_score(df['overall'], df['predicted'], average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# Load Low-Rating Model
low_rating_model_path = '/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model'
low_rating_tokenizer = BertTokenizer.from_pretrained(low_rating_model_path)
low_rating_model = BertForSequenceClassification.from_pretrained(low_rating_model_path)
low_rating_model.to(device)

# Filter Low Ratings
low_rating_df = df[df['predicted'].isin([1, 2])]
low_rating_inputs = low_rating_df['combined_text'].tolist()

# Tokenize low-rating inputs properly
low_rating_tokens = low_rating_tokenizer(
    list(low_rating_df['combined_text']), 
    padding=True, 
    truncation=True, 
    return_tensors="pt", 
    max_length=128
)

# DataLoader for Cause Classification
cause_dataset = TensorDataset(low_rating_tokens['input_ids'], low_rating_tokens['attention_mask'])
cause_dataloader = DataLoader(cause_dataset, batch_size=32)

all_causes = []

low_rating_model.eval()
with torch.no_grad():
    for batch in cause_dataloader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = low_rating_model(input_ids=input_ids, attention_mask=attention_mask)
        causes = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_causes.extend(causes)

# Map numeric cause predictions to category names
cause_mapping = {0: "Damaged", 1: "Expired", 2: "Delivery Issue", 3: "Incorrect", 4: "Quality Issue"}
low_rating_df['cause'] = [cause_mapping[c] for c in all_causes]

# Merge & Save
df.loc[low_rating_df.index, 'cause'] = low_rating_df['cause']
df.to_csv('/home/gridsan/kpower/BERT_for_Amazon_combined/predicted_analysis_with_cause.csv', index=False)

print("Test predictions with causes saved to CSV.")
