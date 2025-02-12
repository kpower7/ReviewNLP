from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the preprocessed DataFrame
df = pd.read_pickle('/home/gridsan/kpower/roBERTa_for_Amazon/processed_reviews__roberta.pkl')

# Combine 'tokens' and 'summarytokens' into a single text input
df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['combined_text'], 
    df['overall'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['overall']
)

# Load the tokenizer and model from the local directory
model_path = '/home/gridsan/kpower/roBERTa_for_Amazon/roberta_pretrained1'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=5, ignore_mismatched_sizes=True, from_tf=True)

# Encode the labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Convert labels to tensor
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Tokenize the text data
def preprocess(data):
    return tokenizer(data, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = preprocess(train_texts.tolist())
test_encodings = preprocess(test_texts.tolist())

# Prepare data in dictionary format
train_data = {
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
}
test_data = {
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
}

# Convert to Dataset format
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

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
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(test_dataset)

# Print the evaluation results
print("Test Set Evaluation Results:", results)


# Define the path where you want to save the trained model and tokenizer
save_path = '/home/gridsan/kpower/roBERTa_for_Amazon/finetuned_model'

# Save the trained model and tokenizer
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Trained model and tokenizer saved to {save_path}.")
print("Trained model saved.")


# Generate predictions on the test set
predictions = trainer.predict(test_dataset)

# Extract predicted labels (class with highest probability)
predicted_labels = predictions.predictions.argmax(axis=-1)

# Convert predicted labels back to original scale
predicted_labels_named = label_encoder.inverse_transform(predicted_labels)

# Save the test set with predictions to a CSV file
test_df = pd.DataFrame({
    'text': test_texts,
    'actual': label_encoder.inverse_transform(test_labels.numpy()),
    'predicted': predicted_labels_named
})
test_df.to_csv('/home/gridsan/kpower/roBERTa_for_Amazon/test_predictions.csv', index=False)
print("Test predictions saved to CSV.")
