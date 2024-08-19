from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to load and tokenize text from a file
def load_and_tokenize(file_path, tokenizer):
    with open(file_path, 'r') as file:
        text = file.read()

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    return inputs

# File paths
file1_path = 'Path to text file'
file2_path = 'Path to text file'

# Load and tokenize the files
inputs1 = load_and_tokenize(file1_path, tokenizer)
inputs2 = load_and_tokenize(file2_path, tokenizer)

# Generate embeddings using BERT
with torch.no_grad():
    embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
    embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

# Calculate cosine similarity between the embeddings
similarity = cosine_similarity(embeddings1, embeddings2)

# Print the similarity score
print(f"Text Similarity: {similarity.item()}")
