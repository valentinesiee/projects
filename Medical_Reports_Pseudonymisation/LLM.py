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
    llm = ChatCohere(cohere_api_key="PmX696EYpvEzZEyUGEuTynIyr1aNvTk5jjsVhZIJ")
    chain = load_qa_chain(llm,chain_type="stuff")
    answer = chain.run(input_documents=docs,question=question)
    return answer

def main():
    load_dotenv()
    st.set_page_config("Medical report summary")
    st.header("Everything about your medical report")
    pdf = st.file_uploader("Upload your PDF Files and Click on the Process Button", type="pdf")

    if pdf is not None:

        #PDF TO TEXT
        texte = pdf_to_text(pdf)

        #TEXT TO CHUNKS
        chunks = text_to_chunks(texte)

        #CHUNKS TO VECTORS
        vectordb = chunks_to_vectors(chunks)
       
        #INFORMATIONS HANDLER
        name = response_to_question("Liste moi juste les noms/prénoms des personnes mentionnées dans le texte",vectordb)
        #print("Name(s) :\n"+name)
        names = name.split(', ')
        #names = [item.strip(', ') for item in names]
        name_to_id = {}
        compteur = 1
        for name in names:
            id = 'ID{}'.format(compteur)
            name_to_id[id] = name
            compteur+=1

        problem = response_to_question("Liste moi juste les maladie/problèmes mentionnées dans le texte",vectordb)
        #print("Problems of the patient :\n"+problem)
        problems = problem.split('\n')
        problems = [item.strip(' -') for item in problems]
        
        treatment = response_to_question("Liste moi juste les traitements/soins mentionnées dans le texte",vectordb)
        #print("Treatments of the patient:\n"+treatment)
        treatments = treatment.split('\n')
        treatments = [item.strip(' -') for item in treatments]

        pseudonomysed_text = response_to_question("Ré-écrit le texte en remplacant juste le(s) nom(s)/prénom(s) avec le(s) id(s) correspondant : "+str(name_to_id),vectordb)
        
        print("Name(s) in the text :\n",names,"\nName(s) / ID(s) table :\n",name_to_id,"\nProblem(s) of the patient :\n",problems,"\nTreatment(s) of the patient :\n",treatments,"\nPseudonymised text :\n",pseudonomysed_text)
            
if __name__ == "__main__":
    main()