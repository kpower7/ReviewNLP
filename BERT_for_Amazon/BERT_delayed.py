import re
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
import torch

# Load the tokenizer and model
model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load your data
df = pd.read_csv("/home/gridsan/kpower/BERT_for_Amazon/Amazon_Grocery_Gourmet_Food_Review_Data.csv")

keywords = [
    "arrive", "delay"
]
# , "late", "deliver", "time", "shipment", "slow", "wait", "schedule", "held", "stuck", "back", "overdue", "missed", "long", "ETA",

# Function to search for keywords in both reviewText and summary
def search_keywords_in_both(text, summary):
    combined_text = f"{text} {summary}"
    pattern = r'(?:' + '|'.join(keywords) + r')'
    return bool(re.search(pattern, combined_text.lower()))

# Apply the function to both columns
df['Contains Keywords'] = df.apply(lambda row: search_keywords_in_both(row['reviewText'], row['summary']), axis=1)
filtered_df = df[df['Contains Keywords']]

# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    rating = torch.argmax(predictions, dim=1).item() + 1  # Add 1 to match your rating system (1-5)
    
    # Classify sentiment based on rating
    if rating <= 2:
        sentiment = 'negative'
    elif rating == 3:
        sentiment = 'neutral'
    else:
        sentiment = 'positive'
    
    return sentiment

# Function to classify sentiment using both reviewText and summary
def classify_sentiment_combined(text, summary):
    combined_text = f"{text} {summary}"
    return classify_sentiment(combined_text)

# Apply sentiment classification to filtered descriptions using both fields
filtered_df['Sentiment'] = filtered_df.apply(lambda row: classify_sentiment_combined(row['reviewText'], row['summary']), axis=1)

#remove duplicates
filtered_df = filtered_df.drop_duplicates()

# Analyze results
negative_reviews = filtered_df[filtered_df['Sentiment'] == 'negative']
print(f"Number of negative reviews due to delayed shipment: {len(negative_reviews)}")

# Save the results to a CSV file
filtered_df.to_csv("/home/gridsan/kpower/BERT_for_Amazon/delayed_shipments_analysis.csv", index=False) 