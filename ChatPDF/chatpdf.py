import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_cohere import ChatCohere

def pdf_to_text(pdfs):
    text = ""
    pdf_reader = PdfReader(pdfs)
    for page in pdf_reader.pages:
        text += page.extract_text()
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
    llm = ChatCohere(cohere_api_key="YOUR API KEY")
    chain = load_qa_chain(llm,chain_type="stuff")
    answer = chain.run(input_documents=docs,question=question)
    return answer

def main():
    load_dotenv()
    st.set_page_config("Ask a question about your pdf")
    st.header("Ask a question about your pdf")
    pdf = st.file_uploader("Upload your PDF Files and Click on the Process Button", type="pdf")

    if pdf is not None:

        #PDF TO TEXT
        texte = pdf_to_text(pdf)

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
