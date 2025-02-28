import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Load the dataset
data_path = '/home/gridsan/kpower/BERT_for_Amazon_LowRating/labelled_negative.csv'
df = pd.read_csv(data_path)

# Load the pre-trained BERT model and tokenizer
model_path = '/home/gridsan/kpower/BERT_for_Amazon_LowRating/bert_pretrained'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=5, ignore_mismatched_sizes=True, from_tf=True)


# Load the model configuration
config = BertConfig.from_pretrained(model_path, num_labels=df['category'].nunique())

# Filter out categories with fewer than 10 occurrences
category_counts = df['category'].value_counts()
valid_categories = category_counts[category_counts >= 10].index
df = df[df['category'].isin(valid_categories)]

# Ensure there are exactly 5 categories remaining
assert df['category'].nunique() == 5, "The number of categories is not equal to 5 after filtering."

# Convert target labels to numeric values
df['category'] = df['category'].astype('category').cat.codes

# Tokenize the input text
inputs = tokenizer(list(df['combined_text']), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Convert target labels to tensor
labels = torch.tensor(df['category'].values, dtype=torch.long)

# Define a custom dataset class
class AmazonDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

# Create the dataset
dataset = AmazonDataset(inputs, labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='/home/gridsan/kpower/BERT_for_Amazon_LowRating/logs',
    logging_steps=200,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model')
tokenizer.save_pretrained('/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model')
