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
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from concurrent.futures import ThreadPoolExecutor
from torch.nn.parallel import DistributedDataParallel as DDP


# Initialize NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

#  Initialize distributed processing
dist.init_process_group(backend='nccl')  # Use 'gloo' if running on CPU

# Load tokenizer & model
model_path = "/home/gridsan/kpower/BERT_for_Amazon_Expanded/bert_pytorch_converted"
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Move model to multiple GPUs
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5)

model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")

model = DDP(model)



# Set multiprocessing parameters
num_proc = min(32, os.cpu_count())

# Define the data directory and file pattern
data_dir = "/home/gridsan/kpower/BERT_for_Amazon_Expanded/data"
file_pattern = "part_{}.jsonl"


# Define training arguments
training_args = TrainingArguments(
    output_dir='/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model',
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  #  Enable mixed precision for speedup
    dataloader_num_workers=4,  #  Helps with parallel data loading
    ddp_find_unused_parameters=False  #  Required for multi-GPU training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Fast text preprocessing function
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

# Process each part file
for i in range(29):
    # Load dataset using pandas
    file_path = os.path.join(data_dir, file_pattern.format(i))
    print(f"Loading data from {file_path}...")
    df = pd.read_json(file_path, lines=True)
    print(f"Data from {file_path} loaded successfully.")

    # Rename columns for consistency
    column_mapping = {
        "reviewText": "text",
        "verified": "verified_purchase",
        "overall": "rating",
        "summary": "title"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Drop rows with invalid ratings
    df = df[df['rating'].isin([1, 2, 3, 4, 5])]

    # Apply preprocessing using pandas
    print(f"Preprocessing data from {file_path}...")
    df['text'] = df['text'].apply(preprocess_text)
    df['title'] = df['title'].apply(preprocess_text)
    print(f"Preprocessing complete for {file_path}.")

    # Combine text fields
    df['combined_text'] = df['text'] + " " + df['title']

    # Encode labels
    label_encoder = LabelEncoder()
    df['rating'] = label_encoder.fit_transform(df['rating'])

    # Tokenize text in parallel with consistent padding
    print(f"Tokenizing data from {file_path}...")
    text_chunks = np.array_split(df['combined_text'].to_list(), num_proc)
    text_chunks = [[text for text in chunk if isinstance(text, str) and text.strip() != ""] for chunk in text_chunks]

    tokenized_inputs = {"input_ids": [], "attention_mask": []}

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        for batch in executor.map(lambda x: tokenizer(x, padding='max_length', truncation=True, return_tensors="pt", max_length=128), text_chunks):
            tokenized_inputs["input_ids"].append(batch["input_ids"])
            tokenized_inputs["attention_mask"].append(batch["attention_mask"])
    print(f"Tokenization complete for {file_path}.")

    # Convert tokenized inputs to PyTorch tensors
    input_ids = torch.cat(tokenized_inputs["input_ids"], dim=0)
    attention_mask = torch.cat(tokenized_inputs["attention_mask"], dim=0)
    labels = torch.tensor(df['rating'].to_list())

    # Ensure all tensors have the same length
    min_length = min(len(input_ids), len(attention_mask), len(labels))
    input_ids = input_ids[:min_length]
    attention_mask = attention_mask[:min_length]
    labels = labels[:min_length]

    # Convert to Hugging Face dataset
    train_encodings = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device)
    }
    train_dataset = Dataset.from_dict(train_encodings)

    # Update Trainer with new dataset
    trainer.train_dataset = train_dataset

    # Train the model on the current part
    print(f"Training on {file_path}...")
    trainer.train()
    print(f"Finished training on {file_path}.")

# Save the trained model and tokenizer after processing all parts
model.save_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model")
tokenizer.save_pretrained("/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model")

print("🚀 Training complete. Model and tokenizer saved.")
