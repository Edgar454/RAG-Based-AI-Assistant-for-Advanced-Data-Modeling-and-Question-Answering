
# AI Assistant for Data Modeling

This project is an **AI-powered assistant** designed to assist data scientists and machine learning practitioners in improving their modeling processes. The assistant offers **suggestions**, **tips**, and **examples of feature engineering** and other techniques to enhance data modeling .

Built on top of a **Retrieval-Augmented Generation (RAG)** system, this assistant combines several advanced technologies:
- **FAISS Vector Database**: Efficiently stores and retrieves contextual information.
- **Sentence Transformer**: Converts text into vector embeddings for similarity search.
- **LLAMA 3.2**: A large language model that drives the conversation and generates responses based on the retrieved context.

By leveraging these components, the assistant can provide actionable insights to help users optimize their data models !


## Features

- **AI-Powered Suggestions**: Receive tips on feature engineering, data preprocessing, and model optimization.
- **Question-Answering (QA)**: Ask questions and get answers based on retrieved data.
- **RAG System**: Combines retrieval of relevant chunks from a knowledge base with generative AI responses.
- **FAISS and LLAMA Integration**: Fast retrieval with FAISS and high-quality responses with LLAMA 3.2.



## Installation

To install the project, you will need to have **Poetry** installed on your system. Poetry is a dependency management tool for Python projects. Follow the steps below:

1. Open your terminal in the directory where the `pyproject.toml` file is located.
2. Install the project dependencies with Poetry:

```bash
poetry install
```
## Run Locally

1. Clone the repository:

```bash
  git clone https://github.com/Edgar454/RAG-Based-AI-Assistant-for-Advanced-Data-Modeling-and-Question-Answering.git

```

2. Navigate to the project directory:

```bash
  cd RAG-Based-AI-Assistant-for-Advanced-Data-Modeling-and-Question-Answering

```

3. Install the required dependencies:

```bash
  poetry install
```

4. Run the assistant:

```bash
  python app.py
```

Once the assistant is running, you can start asking questions and receiving data modeling tips or feature engineering examples to improve your machine learning models.
