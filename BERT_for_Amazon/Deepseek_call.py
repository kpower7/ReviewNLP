import pandas as pd
import time
from openai import OpenAI

# Set your Deepseek API key
# Get your API key from Deepseek
client = OpenAI(
    api_key="",  # Replace with your Deepseek API key
    base_url="https://api.deepseek.com/v1"  # Deepseek API endpoint
)

# File path
file_path = r"C:\Users\k_pow\OneDrive\Documents\MIT\MITx SCM\Spring 2025\SCM256\Project\Archive Files\low_ratings.csv"

# Load dataset
df = pd.read_csv(file_path)

# Sample 300 reviews for testing
df_sample = df.sample(n=300, random_state=42)

# Define categories with shorter labels
categories = {
    "Damaged": "Items arriving with broken seals, crushed boxes, or leaking liquids.",
    "Expired": "Complaints about receiving food past its expiration date or tasting old.",
    "Delivery": "Items arriving late, shipped incorrectly, or exposed to extreme temperatures.",
    "Quality": "Customers disappointed by taste, texture, or quality not matching expectations.",
    "Incorrect": "Receiving the wrong product, incorrect quantity, or missing items."
}

# Function to create a structured prompt
def create_prompt(reviews):
    category_text = "\n".join([f"- {key}: {value}" for key, value in categories.items()])
    prompt = (
        "Classify each of the following customer reviews into one of the five predefined categories below:\n\n"
        f"{category_text}\n\n"
        "Respond with only the category name (one word) for each review.\n\n"
    )
    
    for i, review in enumerate(reviews, start=1):
        prompt += f"{i}. {review}\nCategory: "
    return prompt

# Function to make API call
def get_labels(reviews, model="deepseek-chat"):  
    prompt = create_prompt(reviews)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at classifying customer complaints into predefined categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000  
        )
        result = response.choices[0].message.content.strip()
        return result.split("\n")  
    except Exception as e:
        print(f"API Error: {e}")
        return [None] * len(reviews)  

# Process data in larger batches
batch_size = 500  
results = []

for i in range(0, len(df_sample), batch_size):
    batch = df_sample["combined_text"][i:i+batch_size].tolist()
    labels = get_labels(batch)
    
    # Store results
    for review, label in zip(batch, labels):
        results.append({"combined_text": review, "category": label})
    
    time.sleep(1)  

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("sampled_classified_reviews_deepseek.csv", index=False)

print("Classification complete! Results saved to 'sampled_classified_reviews_deepseek.csv'.")
