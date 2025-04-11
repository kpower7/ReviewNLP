from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bs4 import BeautifulSoup
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text function
def preprocess_text(text):
    # 1. Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # 2. Lowercase
    text = text.lower()
    # 3. Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # 4. Tokenize
    tokens = word_tokenize(text)
    # 5. Remove non-alphabetic tokens and stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # 6. Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Load the dataset
data_path = '/home/gridsan/kpower/BERT_for_Amazon_Expanded/part_0.jsonl'
df = pd.read_json(data_path, lines=True)

# sample for testing
# df = df.sample(n=1000, random_state=42)

# Rename the columns
column_mapping = {
    'reviewText': 'text',
    'verified': 'verified_purchase',
    'overall': 'rating',
    'summary': 'title'
}
df.rename(columns=column_mapping, inplace=True)

# Preprocess the text data
df['text'] = df['text'].apply(preprocess_text)
df['title'] = df['title'].apply(preprocess_text)

# Combine 'text' and 'title' into a single text input
df['combined_text'] = df['text'] + " " + df['title']

# Use all data for training
train_texts = df['combined_text']
train_labels = df['rating']

# Load the tokenizer and model from the local directory
model_path = '/home/gridsan/kpower/BERT_for_Amazon_Expanded/bert_pretrained'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5, ignore_mismatched_sizes=True, from_tf=True)

# Encode the labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

# Convert labels to tensor
train_labels = torch.tensor(train_labels)

# Tokenize the text data
def preprocess(data):
    return tokenizer(data, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = preprocess(train_texts.tolist())

# Prepare data in dictionary format
train_data = {
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
}

# Convert to Dataset format
train_dataset = Dataset.from_dict(train_data)

# Define custom metric computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model',
    evaluation_strategy="no",  # No evaluation since we're using all data for training
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
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model')
tokenizer.save_pretrained('/home/gridsan/kpower/BERT_for_Amazon_Expanded/finetuned_model')

print("Trained model and tokenizer saved.")
