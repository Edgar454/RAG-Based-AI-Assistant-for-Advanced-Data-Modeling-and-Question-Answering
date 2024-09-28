from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import gradio as gr


#Loading the document

loader = PDFPlumberLoader('M:\Youtube_Course\Personnal_projects\LLM_and_NLP\RAG_Research\Documents\Feature Engineering for Machine Learning (Alice Zheng, Amanda Casari) (Z-Library).pdf')
docs = loader.load()

# splitting the document
text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

#initialize the embedding model
embedder = HuggingFaceEmbeddings()

#creation of the vector store
vector = FAISS.from_documents(documents, embedder)


# Input
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retrieved_docs = retriever.invoke("What is binarization ?")


# Definition of the llm
llm = Ollama(model = "llama3.2")


# Definition of the system prompt
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_chain_prompt = PromptTemplate.from_template(prompt)

#Let's put the element together and define the chain
llm_chain = LLMChain(llm= llm,
                     prompt = QA_chain_prompt,
                     callbacks = None,
                     verbose = True)

document_prompt = PromptTemplate(input_variables = ["page_content","source"],
                                 template = "Context:\ncontent:{page_content}\nsource:{source}")


combine_document_chain = StuffDocumentsChain(llm_chain = llm_chain,
                                             document_variable_name = "context",
                                             document_prompt = document_prompt,
                                             callbacks = None)

#Now that the LLM chain and the document are defined let's build our QA complete chain

qa = RetrievalQA(combine_documents_chain = combine_document_chain ,
                 verbose = True ,
                 retriever = retriever,
                 return_source_documents = True)


def respond(question , history):
    return qa(question)['result']

gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to Feature Engineering", container=False, scale=7),
    title="Plant's Chatbot",
    examples=["What is binarization ?", "How to engineer a categorical feature ?"],
    cache_examples=True,
    retry_btn=None,

).launch(share = True)
