import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

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

# Perform sentiment analysis
sentiment_model.eval()
with torch.no_grad():
    sentiment_outputs = sentiment_model(**sentiment_inputs)
    sentiment_predictions = torch.argmax(sentiment_outputs.logits, dim=1)

# Add sentiment predictions to the dataframe
df['predicted'] = sentiment_predictions.numpy()

# Load the LowRating fine-tuned model and tokenizer
low_rating_model_path = '/home/gridsan/kpower/BERT_for_Amazon_LowRating/finetuned_model'
low_rating_tokenizer = BertTokenizer.from_pretrained(low_rating_model_path)
low_rating_model = BertForSequenceClassification.from_pretrained(low_rating_model_path)

# Filter rows with predicted rating of 1 or 2
low_rating_df = df[df['predicted'].isin([1, 2])]

# Tokenize the low rating rows for cause classification
low_rating_inputs = low_rating_tokenizer(list(low_rating_df['combined_text']), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Perform cause classification
low_rating_model.eval()
with torch.no_grad():
    low_rating_outputs = low_rating_model(**low_rating_inputs)
    low_rating_causes = torch.argmax(low_rating_outputs.logits, dim=1)

# Add cause predictions to the low rating dataframe
low_rating_df['cause'] = low_rating_causes.numpy()

# Merge the cause predictions back into the original dataframe
df.loc[low_rating_df.index, 'cause'] = low_rating_df['cause']

# Save the dataframe with sentiment and cause analysis
output_path = '/home/gridsan/kpower/BERT_for_Amazon_combined/predicted_analysis_with_cause.csv'
df.to_csv(output_path, index=False)

print("Test predictions with causes saved to CSV.")
