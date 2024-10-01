import numpy as np
import faiss
import torch
from datasets import load_dataset
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AutoModel, AutoTokenizer

# Load the EURLEX dataset
dataset = load_dataset("eurlex", split="train[:5%]", trust_remote_code=True)

# Check if 'text' or 'document' exists and create a corpus
if "text" in dataset.column_names:
    corpus = dataset['text']
elif "document" in dataset.column_names:
    corpus = dataset['document']
else:
    raise ValueError("Dataset does not contain 'text' or 'document' columns.")

# Load a transformer model for embeddings (e.g., DistilBERT)
embed_model_name = "distilbert-base-uncased"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name)

# Function to get embeddings for the corpus
def get_embeddings(corpus):
    embeddings = []
    for doc in corpus:
        inputs = embed_tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = embed_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Get embeddings for the corpus
embeddings = get_embeddings(corpus)

# Create a FAISS index for the embeddings
d = embeddings.shape[1]  # dimensionality of embeddings
index = faiss.IndexFlatIP(d)  # Inner Product index

# Add embeddings to the index
index.add(embeddings)

# Load the RAG tokenizer and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

# Create a custom retriever using the local FAISS index
class CustomRetriever(RagRetriever):
    def __init__(self, index, tokenizer, corpus):
        self.index = index
        self.tokenizer = tokenizer
        self.corpus = corpus
    
    def retrieve(self, question, n_docs=5):
        question_embedding = get_embeddings([question])
        D, I = self.index.search(question_embedding, n_docs)
        return [self.corpus[i] for i in I[0]], D[0]

retriever = CustomRetriever(index, embed_tokenizer, corpus)

# Sample query
query = "What were the main legal points in the case of freedom of expression vs national security?"
inputs = tokenizer(query, return_tensors="pt")

# Retrieve relevant documents
retrieved_docs, doc_scores = retriever.retrieve(query)

# Prepare context from the corpus using retrieved docs
context_inputs = tokenizer(retrieved_docs, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Generate summary from the retrieved documents
generated_ids = model.generate(
    context_input_ids=context_inputs.input_ids,
    context_attention_mask=context_inputs.attention_mask,
    doc_scores=torch.tensor(doc_scores).unsqueeze(0),
    input_ids=inputs.input_ids,
    num_return_sequences=1,
    num_beams=5,
    max_length=200  # Adjust as needed
)

# Decode and print the summary
summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(f"Query: {query}")
print(f"Summary of the retrieved documents: \n{summary[0]}")
