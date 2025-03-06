# %% [markdown]
# ![image.png](attachment:image.png)
# 
# # **SCM256 Project - Amazon Exploratory Data Analysis**
# 
# This notebook performs exploratory data analysis on Amazon Grocery & Gourmet Food Reviews dataset. The analysis includes:
# 1. Data cleaning and preprocessing
# 2. Basic statistical analysis
# 3. Temporal analysis of reviews
# 4. Text analysis and visualization
# 5. Sentiment distribution analysis
# 
# The final goal is to prepare the data for sentiment analysis using NLP techniques.

# %% [markdown]
# ## Setting up the environment
# 
# We'll import necessary packages for data manipulation, visualization, and text processing.

# %% [markdown]
# ### Import packages

# %%
#general purpose packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
#import tensorflow as tf

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# %% [markdown]
# ### Import data
# 
# Loading the Amazon reviews dataset from local storage.

# %%
df = pd.read_csv('/home/gridsan/kpower/BERT_for_Amazon/Amazon_Grocery_Gourmet_Food_Review_Data.csv')
#df = pd.read_csv('https://www.dropbox.com/scl/fo/n27f2piibmdl5twymhv29/AIOqG0xduoKp-V8O66ZKens?e=1&preview=Amazon_Grocery_Gourmet_Food_Review_Data.csv&rlkey=q4j2a14meio2q4oh6rtutjqlf&st=lsxmt684&dl=1')

# %% [markdown]
# ## Data Exploration and Quality Analysis
# 
# We'll start by examining the dataset structure, checking for missing values, and understanding the basic characteristics of our data.

# %%
df.info()

# %%
df.tail()

# %%
print("DataFrame Shape:", df.shape)              # Rows x Columns
print("\nColumn Names:", df.columns)             # List column names
print("\nInfo:\n")
df.info()                                        # Detailed info (non-nulls, dtypes, etc.)
print("\nStatistical Description (Numeric Columns):\n", df.describe())

# %%
#remove duplicates
df = df.drop_duplicates()

# %%
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Visualize missingness with a heatmap:
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# %%
print("\nUnique values in 'overall' (Ratings):", df['overall'].unique())

# Countplot for the 'overall' column
plt.figure(figsize=(8,5))
sns.countplot(x='overall', data=df, palette='viridis')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# %%
print("\nValue Counts for 'verified':\n", df['verified'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='verified', data=df, palette='viridis')
plt.title("Verified vs Unverified Reviews")
plt.xlabel("Verified Purchase")
plt.ylabel("Count")
plt.show()

# %%
# Convert 'reviewTime' 
df['reviewTime'] = pd.to_datetime(df['reviewTime'], errors='coerce')

# Quick check of distribution over time 
df.set_index('reviewTime', inplace=True)  # set time index to facilitate time-series plots
df.resample('M')['overall'].count().plot(figsize=(10,5))
plt.title("Number of Reviews Over Time (Monthly)")
plt.xlabel("Month")
plt.ylabel("Review Count")
plt.show()

# Reset index if needed for further analysis
df.reset_index(inplace=True)

# %%
df['reviewText'] = df['reviewText'].astype(str)  # Ensure it's string type

# Number of words in each review
df['word_count'] = df['reviewText'].apply(lambda x: len(x.split()))
print("\nWord Count Stats:\n", df['word_count'].describe())

# Distribution of word counts
plt.figure(figsize=(8,5))
sns.histplot(df['word_count'], bins=100, kde=True, color='purple')
plt.title("Distribution of Review Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ## Review Distribution Analysis
# 
# Let's analyze the distribution of reviews across different dimensions:
# 1. Rating distribution
# 2. Verified vs unverified purchases
# 3. Temporal patterns
# 4. Review length analysis

# %%
# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# 1. Rating Distribution
plt.subplot(2, 2, 1)
sns.countplot(x='overall', data=df, palette='viridis')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

# 2. Verified vs Unverified
plt.subplot(2, 2, 2)
verified_pct = df['verified'].value_counts(normalize=True) * 100
plt.pie(verified_pct, labels=['Verified', 'Unverified'], autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title("Verified vs Unverified Reviews")

# 3. Average Rating by Verification Status
plt.subplot(2, 2, 3)
sns.boxplot(x='verified', y='overall', data=df, palette='Set3')
plt.title("Rating Distribution by Verification Status")
plt.xlabel("Verified Purchase")
plt.ylabel("Rating")

# 4. Word Count vs Rating
plt.subplot(2, 2, 4)
sns.boxplot(x='overall', y='word_count', data=df, palette='Set2')
plt.title("Review Length by Rating")
plt.xlabel("Rating")
plt.ylabel("Word Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Temporal Analysis
# 
# Let's analyze how reviews and ratings change over time.

# %%
# Convert reviewTime to datetime if not already
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# Create multiple time-based visualizations
fig = plt.figure(figsize=(15, 10))

# 1. Reviews per month
plt.subplot(2, 2, 1)
df.set_index('reviewTime')['overall'].resample('M').count().plot()
plt.title("Number of Reviews Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Reviews")

# 2. Average rating over time
plt.subplot(2, 2, 2)
df.set_index('reviewTime')['overall'].resample('M').mean().plot()
plt.title("Average Rating Over Time")
plt.xlabel("Date")
plt.ylabel("Average Rating")

# 3. Verified purchase ratio over time
plt.subplot(2, 2, 3)
verified_ratio = df.set_index('reviewTime')['verified'].resample('M').mean()
verified_ratio.plot()
plt.title("Ratio of Verified Purchases Over Time")
plt.xlabel("Date")
plt.ylabel("Ratio of Verified Purchases")

# 4. Average word count over time
plt.subplot(2, 2, 4)
df.set_index('reviewTime')['word_count'].resample('M').mean().plot()
plt.title("Average Review Length Over Time")
plt.xlabel("Date")
plt.ylabel("Average Word Count")

plt.tight_layout()
plt.show()

# Reset index
df.reset_index(drop=True, inplace=True)

# %% [markdown]
# ## Text Analysis and Preprocessing
# 
# We'll now analyze the text content of reviews, including:
# 1. Word frequency analysis
# 2. Word clouds
# 3. Text preprocessing for sentiment analysis

# %%
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# %%
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# %% [markdown]
# Sample for faster processing... will have to batch (ssh into SuperCloud) for full dataset

# %%
# df =df.sample(10000)

# %%
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
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
    return tokens

# Apply preprocessing to reviewText
df['tokens'] = df['reviewText'].astype(str).apply(preprocess_text)
# Apply preprocessing to summary
df['summarytokens'] = df['summary'].astype(str).apply(preprocess_text)

# %%
# Flatten list of all tokens
review_tokens = [word for tokens in df['tokens'] for word in tokens]

# %%
# Frequency distribution
freq_dist = nltk.FreqDist(review_tokens)
common_words = freq_dist.most_common(20)
print("\nTop 20 Frequent Words in review:")
for word, freq in common_words:
    print(word, ":", freq)

# %%
# wordcloud_text = " ".join(review_tokens)

# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
# plt.figure(figsize=(10,5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("Word Cloud of Review Text")
# plt.show()

# %%
summary_tokens = [word for summarytokens in df['summarytokens'] for word in summarytokens]

# %%
# Frequency distribution
freq_dist = nltk.FreqDist(summary_tokens)
common_words = freq_dist.most_common(20)
print("\nTop 20 Frequent Words in summary:")
for word, freq in common_words:
    print(word, ":", freq)

# # %%
# wordcloud_text = " ".join(summary_tokens)

# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
# plt.figure(figsize=(10,5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("Word Cloud of Summary Text")
# plt.show()

# %% [markdown]
# ## Additional Text Analysis

# %%
# Calculate average word count by rating
avg_words_by_rating = df.groupby('overall')['word_count'].mean()
print("\nAverage Word Count by Rating:")
print(avg_words_by_rating)

# Visualization of word count distribution by rating
plt.figure(figsize=(10, 6))
sns.violinplot(x='overall', y='word_count', data=df)
plt.title("Word Count Distribution by Rating")
plt.xlabel("Rating")
plt.ylabel("Word Count")
plt.show()

# %% [markdown]
# ## Save Processed DataFrame
# 
# Save the preprocessed and tokenized dataframe for future sentiment analysis.

# %%
# Save the processed dataframe
processed_df = df[['overall', 'verified', 'reviewTime', 'tokens', 'summarytokens', 'word_count']]
processed_df.to_pickle('processed_reviews.pkl')
print("Processed dataframe saved as 'processed_reviews.pkl'")
