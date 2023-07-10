import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

# Set Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'Enter your key here'

# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """
You are an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's question
Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create the Streamlit app
def main():
    st.title("Falcon LLM")
    st.markdown("""
    Welcome to the Falcon Language Model (LLM). This AI assistant is designed to provide 
    helpful, detailed, and polite answers to your questions. Feel free to ask anything! 
    """)

    # Add sidebar content
    st.sidebar.title("About Falcon LLM")
    st.sidebar.markdown("""
    Falcon LLM is a sophisticated language model built on the top of HuggingFaceHub. It's designed
    to understand complex human language and provide meaningful and insightful responses.
    """)

    # Add your social media links to the sidebar
    st.sidebar.markdown('**Connect**')
    st.sidebar.markdown('[GitHub](https://github.com/datamokotow)')
    st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/rutvikacharya/)')
    st.sidebar.markdown('[Twitter](https://twitter.com/datamokotow)')

    # Initialize messages if not in the session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if question := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            # Display the bot thinking...
            message_placeholder = st.empty()
            full_response = ""
            # Generate the response
            with st.spinner("Generating Answer..."):
                response = llm_chain.run(question)
            full_response += response
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
