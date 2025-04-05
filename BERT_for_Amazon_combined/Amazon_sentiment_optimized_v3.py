import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor

# Import NLP preprocessing tools
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the original dataset
df = pd.read_csv('/home/gridsan/kpower/BERT_for_Amazon/Amazon_Grocery_Gourmet_Food_Review_Data.csv')

# Remove duplicates
df = df.drop_duplicates()

# Convert 'reviewTime' to datetime format
df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')

# Ensure 'reviewText' is a string type
df['reviewText'] = df['reviewText'].astype(str)

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Lowercase text
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Apply text preprocessing to review text & summary
df['tokens'] = df['reviewText'].apply(preprocess_text)
df['summarytokens'] = df['summary'].astype(str).apply(preprocess_text)

# Combine 'tokens' and 'summarytokens' into a single text input for tokenization
df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

# Load Model & Tokenizer
sentiment_model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)

# Move Model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)

# Tokenization in Smaller Batches to Avoid OOM
batch_size = 10000
tokenized_results = []

for i in range(0, len(df), batch_size):
    batch_texts = df['combined_text'].tolist()[i : i + batch_size]
    tokenized_results.append(sentiment_tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128))

sentiment_inputs = {
    'input_ids': torch.cat([r['input_ids'] for r in tokenized_results], dim=0).to(device),
    'attention_mask': torch.cat([r['attention_mask'] for r in tokenized_results], dim=0).to(device)
}

# DataLoader for Batched Inference
dataset = TensorDataset(sentiment_inputs['input_ids'], sentiment_inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=32)

all_predictions = []

sentiment_model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask = [t.to(device) for t in batch]
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

# Tokenize low-rating inputs
low_rating_tokens = low_rating_tokenizer(
    low_rating_df['combined_text'].tolist(),
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
        input_ids, attention_mask = [t.to(device) for t in batch]
        outputs = low_rating_model(input_ids=input_ids, attention_mask=attention_mask)
        causes = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        all_causes.extend(causes)

cause_mapping = {0: "Damaged", 1: "Expired", 2: "Delivery Issue", 3: "Incorrect", 4: "Quality Issue"}
low_rating_df['cause'] = [cause_mapping[c] for c in all_causes]

df.loc[low_rating_df.index, 'cause'] = low_rating_df['cause']
df.to_csv('/home/gridsan/kpower/BERT_for_Amazon_combined/predicted_analysis_with_cause_v3.csv', index=False)

print("Test predictions with causes saved to CSV.")
