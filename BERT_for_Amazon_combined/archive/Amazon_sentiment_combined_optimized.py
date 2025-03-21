import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

def setup(rank, world_size):
    # Use SLURM environment variables for setting up the process group
    dist.init_process_group(
        backend='nccl',  # Use 'nccl' for GPU, 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    # Load the preprocessed DataFrame
    df = pd.read_pickle('/home/gridsan/kpower/BERT_for_Amazon/processed_reviews.pkl')

    # Combine 'tokens' and 'summarytokens' into a single text input
    df['combined_text'] = df['tokens'].apply(lambda x: " ".join(x)) + " " + df['summarytokens'].apply(lambda x: " ".join(x))

    # Load the pre-trained sentiment analysis model and tokenizer
    sentiment_model_path = '/home/gridsan/kpower/BERT_for_Amazon/finetuned_model'
    sentiment_tokenizer = BertTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path).to(rank)
    sentiment_model = DDP(sentiment_model, device_ids=[rank])

    # Tokenize the test set for sentiment analysis
    sentiment_inputs = sentiment_tokenizer(list(df['combined_text']), padding='max_length', truncation=True, return_tensors="pt", max_length=128)

    # Create a DataLoader for batch processing
    dataset = TensorDataset(sentiment_inputs['input_ids'], sentiment_inputs['attention_mask'])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Perform sentiment analysis
    sentiment_model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(rank)
            attention_mask = attention_mask.to(rank)
            outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1) + 1  # Adjust to range 1-5
            all_predictions.extend(predictions.cpu().numpy())

    # Gather predictions from all processes
    all_predictions = torch.tensor(all_predictions).to(rank)
    dist.all_gather(all_predictions, all_predictions)

    if rank == 0:
        # Add sentiment predictions to the dataframe
        df['predicted'] = all_predictions.cpu().numpy()

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(df['overall'], df['predicted'])
        f1 = f1_score(df['overall'], df['predicted'], average='weighted')

        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Save the dataframe with sentiment and cause analysis
        output_path = '/home/gridsan/kpower/BERT_for_Amazon_combined/predicted_analysis_with_cause.csv'
        df.to_csv(output_path, index=False)

        print("Test predictions with causes saved to CSV.")

    cleanup()

if __name__ == "__main__":
    world_size = int(os.environ['SLURM_NTASKS'])  # Use SLURM's environment variable
    rank = int(os.environ['SLURM_PROCID'])  # Use SLURM's environment variable
    main(rank, world_size)
