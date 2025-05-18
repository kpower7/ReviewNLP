import os

def split_jsonl(input_file, output_dir, lines_per_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as infile:
        file_count = 0
        line_count = 0
        outfile = None

        for line in infile:
            if line_count % lines_per_file == 0:
                if outfile:
                    outfile.close()
                outfile = open(os.path.join(output_dir, f'chunk_{file_count}.jsonl'), 'w')
                file_count += 1

            outfile.write(line)
            line_count += 1

        if outfile:
            outfile.close()

    print(f"Split into {file_count} files.")

# Example usage
input_file = '/home/gridsan/kpower/BERT_for_Amazon_Expanded/Grocery_and_Gourmet_Food.jsonl'
output_dir = '/home/gridsan/kpower/BERT_for_Amazon_Expanded/daily_trained'
lines_per_file = 1000000  # Adjust the number of lines per chunk as needed

split_jsonl(input_file, output_dir, lines_per_file) 