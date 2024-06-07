from langchain.document_loaders import UnstructuredURLLoader
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_cohere import ChatCohere

def url_to_text(URLs):
    loaders=UnstructuredURLLoader(urls=URLs)
    text_list=loaders.load()
    text = "".join([doc.page_content for doc in text_list])
    return text 

def text_to_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def chunks_to_vectors(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

def response_to_question(question,vectordb):
    docs = vectordb.similarity_search(question)
    llm = ChatCohere(cohere_api_key="PmX696EYpvEzZEyUGEuTynIyr1aNvTk5jjsVhZIJ")
    chain = load_qa_chain(llm,chain_type="stuff")
    answer = chain.run(input_documents=docs,question=question)
    return answer

def main():
    load_dotenv()
    st.set_page_config("Ask a question about the content of the website you want")
    st.header("Ask a question about the content of the website you want")
    url = st.text_input("Enter the URL of the website")

    if url is not None:

        #URL TO TEXT
        texte = url_to_text([url])

        #TEXT TO CHUNKS
        chunks = text_to_chunks(texte)
     
        #CHUNKS TO VECTORS
        vectordb = chunks_to_vectors(chunks)
       
        #QUESTION HANDLER
        question = st.text_input("Ask the question you want")
        if question:
            answer = response_to_question(question,vectordb)
            st.success(answer)
        
if __name__ == "__main__":
    main()