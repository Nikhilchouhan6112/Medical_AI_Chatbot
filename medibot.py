import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceInferenceAPIEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN=os.environ.get("HF_TOKEN")

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, 
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    # Create the endpoint
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.5,
    )
    # Wrap it in ChatHuggingFace for better compatibility
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """Answer the question using ONLY relevant information from the context.

Context: {context}

Question: {question}

Instructions:
- If the context contains the answer, provide 2-3 bullet points
- If the context does NOT directly answer the question, say "I don't have specific information about that in my knowledge base"
- Do NOT provide tangentially related information
- Do NOT list organizations or resources unless specifically asked

Answer:"""
        
        HUGGINGFACE_REPO_ID="HuggingFaceH4/zephyr-7b-beta"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':2}),  # Reduced from 3 to 2
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            
            # Format source documents nicely
            sources_text = "\n\n**Sources:**\n"
            for i, doc in enumerate(source_documents, 1):
                page = doc.metadata.get('page', 'Unknown')
                source_file = doc.metadata.get('source', 'Unknown').split('\\')[-1]
                sources_text += f"\n{i}. {source_file} (Page {page})"
            
            result_to_show = result + sources_text
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            import traceback
            error_details = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            st.error(error_details)
            print(error_details)  # Also print to console

if __name__ == "__main__":
    main()
