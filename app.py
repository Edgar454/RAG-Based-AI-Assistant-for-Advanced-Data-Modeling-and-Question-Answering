import streamlit as st
import time
from rag_system import RAG_System

# Initialize the RAG system only if it hasn't been done already
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAG_System(
        db_path="faiss.db",
        llm_model="qwen2.5:1.5b"
    )

def respond_with_error_handling(question, history=[]):
    try:
        # Combine the history and the current question into a single context string
        response = st.session_state.rag_system.respond(question , history)
        print(response)
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    except Exception as e:
        return f"Error: {str(e)}"


# Track if the assistant is loading
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
    
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

st.title("FeatGuru ⚙")
st.markdown("**Your Guide to Feature Engineering – Ask Anything!**")


st.sidebar.subheader("Contact Me")

# Add clickable links for GitHub, LinkedIn, and email
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Edgar454)")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/edgar-peggy-meva-a-16a93a267/)")
st.sidebar.markdown("[📧 Email me](mailto:mevaed4@gmail.com)")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input, disable when loading
if prompt := st.chat_input("Ask me anything about feature engineering! 😄", disabled=st.session_state.is_loading):
    st.session_state.current_prompt = prompt 

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    

    # Set loading state to True to disable input
    st.session_state.is_loading = True

    # Trigger a rerun to immediately disable the input field
    st.rerun()

# Handle assistant's response (after rerun)
if st.session_state.is_loading:

    # Prepare the history for the RAG system
    history = [msg["content"] for msg in st.session_state.messages]

    with st.chat_message("assistant"):
        # processing the input
        response = st.write_stream(respond_with_error_handling(st.session_state.current_prompt,history))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": "".join(response)})

    # Reset loading state after processing and rerun to enable input
    st.session_state.is_loading = False
    st.session_state.current_prompt = ""
    st.rerun()




