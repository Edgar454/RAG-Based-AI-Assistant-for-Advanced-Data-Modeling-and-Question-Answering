import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import pandas as pd
from tqdm.auto import tqdm
from rag_system import RAG_System
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Initialize ROUGE and BLEU evaluation tools
rouge = Rouge()
smooth = SmoothingFunction()

# Function to load questions and answers from a text file
def load_questions_and_answers(file_path):
    qa_pairs = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        question = None
        for line in lines:
            line = line.strip()
            if line.startswith("Q:"):
                question = line[3:].strip()
            elif line.startswith("A:") and question:
                answer = line[3:].strip()
                qa_pairs[question] = answer
    return qa_pairs

# Load questions and answers from the text file
qa_file_path = 'questions_and_answers.txt'
qa_pairs = load_questions_and_answers(qa_file_path)

# Define evaluation metrics
rouge_scores = []
bleu_scores = []
retrieval_accuracy = []
response_times = []

# Function to evaluate the RAG system
def evaluate_rag(eval_path = "eval.csv"):
    rag_system = RAG_System(db_path="M:/Youtube_Course/Personnal_projects/LLM_and_NLP/RAG_Research/faiss.db",llm_model = "qwen2.5:1.5b" )

    for question, ground_truth in tqdm(qa_pairs.items()):
        start_time = time.time()
        response = rag_system.respond(question, "")
        end_time = time.time()
        
        print(f"Question: {question}")
        print(f"RAG Response: {response}")

        #Response time evaluation
        response_time = end_time - start_time
        response_times.append(response_time)
        
        # ROUGE score evaluation
        rouge_score = rouge.get_scores(response, ground_truth)
        rouge_scores.append(rouge_score[0]["rouge-l"]["f"])

        # BLEU score evaluation
        response_tokens = response.lower().split()
        ground_truth_tokens = ground_truth.lower().split()
        bleu_score = sentence_bleu([ground_truth_tokens], response_tokens, smoothing_function=smooth.method1)
        bleu_scores.append(bleu_score)

        # Perform FAISS search for retrieval accuracy
        top_k_retrieved_chunks = rag_system.vectordb.search(question,search_type="similarity", k=3)  # Search with top-k results
        retrieved_texts = [chunk.page_content for chunk in top_k_retrieved_chunks]

        if any(ground_truth in chunk_text for chunk_text in retrieved_texts):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)

    # Calculate and print overall metrics
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    retrieval_accuracy_score = sum(retrieval_accuracy) / len(retrieval_accuracy)
    avg_response_time = sum(response_times) / len(rouge_scores)


    eval_dict = {"model":rag_system.llm_model,
            "avg_response_time":avg_response_time,
            "rouge_score":avg_rouge_score,
            "bleu_score":avg_bleu_score,
            "retrieval_accuracy_score":retrieval_accuracy_score,}
    
    eval_df = pd.DataFrame([eval_dict])

    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        df = pd.concat([df, eval_df], axis=0)
    else:
        df = eval_df

    # Write the updated dataframe to the CSV file
    df.to_csv(eval_path, index=False)
    
    
    print(f"Average ROUGE-L Score: {avg_rouge_score:.4f}")
    print(f"Average BLEU Score: {avg_bleu_score:.4f}")
    print(f"Retrieval Accuracy: {retrieval_accuracy_score:.4f}")

if __name__ == "__main__":
    evaluate_rag()
