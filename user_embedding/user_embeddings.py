from sentence_transformers import SentenceTransformer
import time
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
tic = time.time()
#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)
toc = time.time() - tic
print(toc)
#Print the embeddings

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

