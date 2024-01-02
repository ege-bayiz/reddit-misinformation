from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
from pprint import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentences(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

dataset = load_dataset('jinaai/negation-dataset')
df_train = pd.DataFrame(dataset['train'])
print(df_train.columns)


# Sentences we want sentence embeddings for
anchor_list = df_train['anchor'].tolist()

anchor_embedding = (encode_sentences(df_train['anchor'].tolist()))
entailment_embedding = (encode_sentences(df_train['entailment'].tolist()))
negative_embedding = (encode_sentences(df_train['negative'].tolist()))

# Save embeddings
torch.save(anchor_embedding, 'user_embedding/datasets/anchor_train.pkl')
torch.save(entailment_embedding, 'user_embedding/datasets/entailment_train.pkl')
torch.save(negative_embedding, 'user_embedding/datasets/negative_train.pkl')

## Test Embeddings
df_test = pd.DataFrame(dataset['test'])
print(df_test.columns)

# Sentences we want sentence embeddings for
anchor_list = df_test['anchor'].tolist()

anchor_embedding = (encode_sentences(df_test['anchor'].tolist()))
entailment_embedding = (encode_sentences(df_test['entailment'].tolist()))
negative_embedding = (encode_sentences(df_test['negative'].tolist()))

# Save embeddings
torch.save(anchor_embedding, 'user_embedding/datasets/anchor_test.pkl')
torch.save(entailment_embedding, 'user_embedding/datasets/entailment_test.pkl')
torch.save(negative_embedding, 'user_embedding/datasets/negative_test.pkl')



