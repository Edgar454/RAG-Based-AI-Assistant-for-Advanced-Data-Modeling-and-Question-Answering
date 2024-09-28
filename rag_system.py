import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import pickle  # For metadata storage

class RAG_System():
    def __init__(self, data_directory, db_path):
        self.data_dir = data_directory
        self.db_path = db_path
        self.embedding_model = "llama3.2"
        self.llm_model = "llama3.2:1b"

        # Load documents
        pages = self._load_documents()
        self.chunks = self._document_splitter(pages)
        self.chunks = self._get_chunk_ids(self.chunks)

        # Initialize the vector database
        print('Initializing the vector database')
        self.vectordb = self._initialize_vectorDB()
        print('Initialization complete')

        self.model = Ollama(model=self.llm_model)

        self.prompt_template = """1. Use the following pieces of context to answer the question at the end.
                                   2. If you don't know the answer, just say "I don't know" but don't make up an answer.
                                   3. Keep the answer crisp and limited to 3-4 sentences.
                                   Context: {context}
                                   Question: {question}
                                   Helpful Answer:"""

        self._setup_collection()
        self.build_the_chain()

    def _setup_collection(self):
        # Load existing metadata
        existing_metadata = self._load_metadata()

        # Get existing IDs
        ids_in_db = set(existing_metadata.keys())
        print(f'Number of existing ids in db: {len(ids_in_db)}')

        # Filter out chunks already in the FAISS vector store
        chunks_to_add = [i for i in self.chunks if i.metadata.get('chunk_id') not in ids_in_db]

        if chunks_to_add:
            # Generate embeddings for the chunks that need to be added
            #embeddings = self._get_embedding_func()  # Get the embedding function

            # Generate embeddings for the chunks
            #vectors_to_add = [embeddings.embed_query(i.page_content) for i in chunks_to_add]  # Compute embeddings
            #ids_to_add = [i.metadata['chunk_id'] for i in chunks_to_add]
            #metadata_to_add = [i.metadata for i in chunks_to_add]

            # Add new vectors to FAISS
            self.vectordb.add_documents(chunks_to_add)
            print(f"Added {len(chunks_to_add)} records to the FAISS DB")

        # Save the updated FAISS index
        self.vectordb.save_local(self.db_path)


    def _get_chunk_ids(self, chunks):
        prev_page_id = None
        curr_page_index = 0  # Initialize the page index
        for i in chunks:
            src = i.metadata.get("source")
            page = i.metadata.get("page")
            curr_page_id = f"{page}_{src}"

            if curr_page_id == prev_page_id:
                curr_page_index += 1
            else:
                curr_page_index = 0
            
            # Final ID of the chunk
            curr_chunk_id = f"{curr_page_id}_{curr_page_index}"
            prev_page_id = curr_page_id
            i.metadata["chunk_id"] = curr_chunk_id
        return chunks
    
    def _intialize_retriever(self):
        retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever
    
    def build_the_chain(self):
        QA_chain_prompt = PromptTemplate.from_template(self.prompt_template)
        retriever = self._intialize_retriever()

        llm_chain = LLMChain(llm= self.model,
                             prompt = QA_chain_prompt,
                             callbacks = None,
                             verbose = True)
        document_prompt = PromptTemplate(input_variables = ["page_content","source"],
                                         template = "Context:\ncontent:{page_content}\nsource:{source}")
        
        combine_document_chain = StuffDocumentsChain(llm_chain = llm_chain,
                                                     document_variable_name = "context",
                                                     document_prompt = document_prompt,
                                                     callbacks = None)
        self.qa = RetrievalQA(combine_documents_chain = combine_document_chain,
                              verbose = True,
                              retriever = retriever,
                              return_source_documents = True)
        
    
    def respond(self, question, history):
        return self.qa(question)['result']

    def _load_documents(self):
        loader = PyPDFDirectoryLoader(self.data_dir)
        pages = loader.load()
        return pages
    
    def _document_splitter(self, documents):
        splitter = SemanticChunker(self._get_embedding_func())
        chunks = splitter.split_documents(documents)
        return self._get_chunk_ids(chunks)  # Get chunk IDs after splitting
    
    def _get_embedding_func(self):
        embeddings = HuggingFaceEmbeddings()
        return embeddings
    
    def _initialize_vectorDB(self):
        if os.path.exists(self.db_path):
            # Load the existing FAISS vector store if it exists
            print(f"Loading existing FAISS vector store from {self.db_path}")
            vector_store = FAISS.load_local(self.db_path, self._get_embedding_func() , allow_dangerous_deserialization=True)
        else:
            # Create a new FAISS vector store from the current chunks
            print(f"No existing vector store found. Creating a new one.")
            vector_store = FAISS.from_documents(self.chunks, self._get_embedding_func())
        return vector_store

    def _load_metadata(self):
        metadata_path = self.db_path + "_metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_metadata(self, metadata):
        metadata_path = self.db_path + "_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
