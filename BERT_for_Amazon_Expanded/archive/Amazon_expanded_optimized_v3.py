import os
import re
import string
import torch
import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor


#----------------------------------------------------------------------------------------
#IMPORTANT:
#change each day:
#data_path = "/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/chunk_2.jsonl"
#day_number = 2  # Change each day
#----------------------------------------------------------------------------------------

# ✅ Use local NLTK data to avoid download issues
# nltk.data.path.append('/home/gridsan/kpower/nltk_data')  # Use manually downloaded NLTK data
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Set multiprocessing parameters
num_proc = min(32, os.cpu_count())

# ✅ Load dataset using pandas

data_path = "/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/chunk_2.jsonl"
df = pd.read_json(data_path, lines=True)
print("Data loaded successfully.")

# ✅ Sample 1000 rows for testing
# df = df.sample(n=1000, random_state=42)

# ✅ Rename columns for consistency
column_mapping = {
    "reviewText": "text",
    "verified": "verified_purchase",
    "overall": "rating",
    "summary": "title"
}
df.rename(columns=column_mapping, inplace=True)

# ✅ Drop rows with invalid ratings
df = df[df['rating'].isin([1, 2, 3, 4, 5])]

# ✅ Fast text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Tokenize and filter words
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# ✅ Apply preprocessing using pandas
df['text'] = df['text'].apply(preprocess_text)
df['title'] = df['title'].apply(preprocess_text)

# ✅ Combine text fields
df['combined_text'] = df['text'] + " " + df['title']

# Filter out rows with missing 'combined_text' or 'rating'
df = df.dropna(subset=['combined_text', 'rating'])

# Ensure 'combined_text' is not empty
df = df[df['combined_text'].str.strip() != ""]

# ✅ Load tokenizer & model
model_path = "/home/gridsan/kpower/BERT_for_Amazon_Expanded/bert_pretrained"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5, from_tf=True).to(device)

# ✅ Encode labels
label_encoder = LabelEncoder()
df['rating'] = label_encoder.fit_transform(df['rating'])

# ✅ Tokenize text in parallel with consistent padding
text_chunks = np.array_split(df['combined_text'].to_list(), num_proc)
text_chunks = [[text for text in chunk if isinstance(text, str) and text.strip() != ""] for chunk in text_chunks]

tokenized_inputs = {"input_ids": [], "attention_mask": []}

with ThreadPoolExecutor(max_workers=num_proc) as executor:
    for batch in executor.map(lambda x: tokenizer(x, padding='max_length', truncation=True, return_tensors="pt", max_length=128), text_chunks):
        tokenized_inputs["input_ids"].append(batch["input_ids"])
        tokenized_inputs["attention_mask"].append(batch["attention_mask"])

# Convert tokenized inputs to PyTorch tensors
train_encodings = {
    "input_ids": torch.cat(tokenized_inputs["input_ids"], dim=0).to(device),
    "attention_mask": torch.cat(tokenized_inputs["attention_mask"], dim=0).to(device),
    "labels": torch.tensor(df['rating'].to_list()).to(device)
}

# Convert to Hugging Face dataset
train_dataset = Dataset.from_dict(train_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/finetuned_model',
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    save_strategy="epoch",
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)


# We define a "day number" so we can store/retrieve each day's checkpoint separately
day_number = 2  # Change each day

# Build directory names for easy reading
previous_checkpoint_dir = f"/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/checkpoint_day_{day_number - 1}"
current_checkpoint_dir = f"/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/checkpoint_day_{day_number}"

# If a checkpoint from previous day exists, resume from it; else start fresh
if os.path.isdir(previous_checkpoint_dir):
    print(f"Resuming from checkpoint: {previous_checkpoint_dir}")
    trainer.train(resume_from_checkpoint=previous_checkpoint_dir)
else:
    print("No checkpoint found, starting from scratch.")
    trainer.train()

# After training, save the entire Trainer checkpoint (model + optimizer + scheduler)
trainer.save_model(current_checkpoint_dir)  # includes model weights
trainer.save_state()                        # includes optimizer/scheduler/training state
print(f"Saved full checkpoint to {current_checkpoint_dir}")

# Optionally, also save a "final" model folder (trainer.save_model() already does this,
# but if you want a single consistent path each time, you can do it again):
model.save_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/finetuned_model")
tokenizer.save_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained/finetuned_model")

print("🚀 Training complete. Full Trainer checkpoint saved.")