
# Custom Retrieval-Augmented Generation (RAG) System

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** system using **FAISS**, **Transformers**, and the **EURLEX dataset**. It enables:

1. **Efficient Document Retrieval**: Find relevant documents using embeddings and FAISS indexing.
2. **Summarization of Retrieved Documents**: Generate summaries using a pre-trained RAG model.
3. **Custom Retriever Implementation**: Use a custom FAISS-based retriever to handle document queries.

---

## Features

- **Dataset**: Uses the EURLEX legal dataset for training and testing.
- **Embedding Model**: DistilBERT is used to create document embeddings.
- **FAISS Indexing**: Efficient similarity search for document retrieval.
- **RAG Model**: Summarizes retrieved documents based on a query.
- **Custom Retriever**: Implements a custom retriever class using FAISS.

---

## Workflow

### 1. Load Dataset
- Load the EURLEX dataset and extract the `text` column as the corpus.

### 2. Generate Embeddings
- Use DistilBERT to create embeddings for all documents in the corpus.

### 3. Index Embeddings with FAISS
- Build a FAISS index for the embeddings, enabling fast similarity search.

### 4. Create Custom Retriever
- Extend `RagRetriever` to implement a custom FAISS-based document retriever.

### 5. Query and Summarization
- Retrieve documents based on a query and generate a summary using the RAG model.

---

## How to Use

### Prerequisites

- Python 3.8+
- Libraries: `transformers`, `datasets`, `faiss`, `torch`, `numpy`

Install required libraries:
```bash
pip install transformers datasets faiss-cpu torch numpy
```

### Run the Code

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Execute the script:
   ```bash
   python main.py
   ```

3. Example output:

   **Query:** `"What were the main legal points in the case of freedom of expression vs national security?"`

   **Retrieved Documents:**
   - "Legal frameworks protecting freedom of speech in the EU."
   - "Cases balancing public safety and individual rights."

   **Generated Summary:**
   - "The case highlights the EU's balancing act between freedom of expression and national security, emphasizing proportionality and necessity in restricting speech."

---

## File Structure

```
.
├── main.py          # Main script for running the RAG system
├── README.md        # Documentation file (this file)
├── requirements.txt # List of required libraries
```

---

## Example Query Results

### Input Query
```text
"What were the main legal points in the case of freedom of expression vs national security?"
```

### Output
- **Retrieved Documents:**
  - "Legal frameworks protecting freedom of speech in the EU."
  - "Cases balancing public safety and individual rights."

- **Generated Summary:**
  - "The case highlights the EU's balancing act between freedom of expression and national security, emphasizing proportionality and necessity in restricting speech."

---

## Future Improvements

- **Interactive Web Interface**: Add a front-end for user interaction.
- **Better Embedding Models**: Experiment with advanced models like BERT or RoBERTa.
- **Support for Larger Datasets**: Optimize memory usage for larger corpora.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
