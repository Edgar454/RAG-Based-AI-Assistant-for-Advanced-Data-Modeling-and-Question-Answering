import gradio as gr
from rag_system import RAG_System

rag_system = RAG_System(data_directory = r"M:/Youtube_Course/Personnal_projects/LLM_and_NLP/RAG_Research/Documents",
                        db_path = "faiss.db")

def respond_with_error_handling(question, history):
    try:
        response = rag_system.respond(question, history)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    gr.ChatInterface(
    respond_with_error_handling,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me question related to Feature Engineering", container=False, scale=7),
    title="Plant's Chatbot",
    examples=["What is binarization ?", "How to engineer a categorical feature ?"],
    cache_examples=True,
    retry_btn=None, ).launch(share = True)