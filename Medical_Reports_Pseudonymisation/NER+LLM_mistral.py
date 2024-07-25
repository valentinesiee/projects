from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_cohere import ChatCohere
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertConfig
from transformers import pipeline
import fitz
import re
import streamlit as st 
import ast
from fpdf import FPDF
import os
from gliner import GLiNER
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from unidecode import unidecode
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage





def pdf_to_text(pdfs):
    """
    Extraire le contenu d'un pdf

    pdfs (file) : PDF que l'utilisateur dépose

    Returns:
        string: Le contenu textuel du PDF
    """
    text = ""
    pdf_reader = PdfReader(pdfs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text    

def text_to_chunks(text):
    """
    Diviser le texte en chunks

    text (string) : Contenu textuel du PDF

    Returns:
        liste: Le contenu textuel divisé en chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def chunks_to_vectors(chunks):
    """
    Calculer le vecteur associé a chaque chunk et les stocker dans une base de données

    chunks (liste) : Le PDF en chunks

    Returns:
        liste: Les vecteurs
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

def response_to_question(question):
    """
    Réponse obtenue en posant une question au LLM

    question (string) : La question que l'on pose au LLM
    vectordb(list) : Base de connaisance qui va permettre au LLM de répondre

    Returns:
        string: La réponse à la question
    """
    # docs = vectordb.similarity_search(question)
    # llm = ChatCohere(cohere_api_key="WAdnOK7MEVQIzXktdZzGEYywOB89J5LMUnO0a24O")
    # chain = load_qa_chain(llm,chain_type="stuff")
    # answer = chain.run(input_documents=docs,question=question)
    

    client_azure = VLLMOpenAI(
    openai_api_base="https://stagiaires.llm.iagen-ov.fr/v1",
    openai_api_key="E5A29DFB-D08C-4C09-BE8F-5231616CDDAD",
    model_name="mistralai/Mixtral-8X7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=7000,
    )


    return client_azure.invoke(question)



def extract_text_from_pdf(pdf_path):
    """
    Extraire le contenu d'un pdf

    pdf_path (string) : Chemin relatif ou absolu menant au PDF

    Returns:
        string: Le contenu textuel du PDF
    """
    # with fitz.open(pdf_path) as pdf_document:
    #     text = ""
    #     for page_number in range(pdf_document.page_count):
    #         page = pdf_document.load_page(page_number)
    #         page_text = page.get_text()
    #         text += page_text
    # return text

    pdf = PyMuPDFLoader(pdf_path.name).load()
    extracted_text = ""
    for document in pdf:
        content = document.page_content
        extracted_text += content
    return extracted_text

def string_to_list(problems_str):
    """
    Créer une liste à partir d'une string

    problems_str (string) : String que l'on veut transformer en liste

    Returns:
        liste: Le contenu de la string dans une liste
    """
    cleaned_str = problems_str.strip()
    try:
        problems_list = ast.literal_eval(cleaned_str)
        return problems_list
    except (ValueError, SyntaxError):
        st.error("The input string is not in the correct format.")
        return []
        
def standardize_dates(text):
    """
    Standardizer les dates

    text (string) : String que l'on veut standardizer

    Returns:
        string: Le contenu de la string avec les dates standardizées
    """
    date_patterns = [
        (r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', '%d/%m/%Y'),
        (r'\b(\d{2,4})[/\-](\d{1,2})[/\-](\d{1,2})\b', '%Y/%m/%d'),
        (r'\b(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{2,4})\b', '%d %b %Y')
    ]
    
    for pattern, date_format in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                date_str = ' '.join(match)
                standardized_date = datetime.strptime(date_str, date_format).strftime('%Y-%m-%d')
                text = text.replace(date_str, standardized_date)
            except ValueError:
                continue

    return text

def clean_special_characters(text):
    """
    Remplacer les caractères spéciaux d'une string

    text (string) : String que l'on veut clean

    Returns:
        string: Le contenu de la string clean
    """
    special_chars = {
        '\n': ' ',  
        '\t': ' ',  
    }

    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def segment_text(text):
    """
    Créer des segments d'une string

    text (string) : String que l'on veut segmenter

    Returns:
        string: Le contenu de la string segmenté et clean
    """
    segments = text.split('\n\n')
    segments = [clean_special_characters(segment) for segment in segments]
    
    return segments

def preprocess_text(text):
    """
    Utiliser les fonctions de preprocess sur une string

    text (string) : String que l'on veut traiter

    Returns:
        string: Le contenu de la string traité
    """
    text = standardize_dates(text)
    text = clean_special_characters(text)
    segments = segment_text(text)
    return segments

def main(): 
    #Configuration de l'interface streamlit
    st.set_page_config("Medical report summary")

    st.markdown(
        """
        <style>
        /* Applique la couleur de fond blanche à la barre latérale */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative; /* Permet de positionner le logo par rapport à la barre latérale */
        }
        .custom-container {
            border: 2px solid #ff4b4b; /* Changer l'épaisseur de la bordure et la couleur */
            padding: 10px;
            border-radius: 5px;
        }
        /* Applique la couleur de fond rouge à l'application principale */
        .stApp {
            background-color: #E35656;
        }

        /* Rendre le titre noir */
        .css-10trblm {  /* Remplacer par la classe appropriée pour le titre, si nécessaire */
            color: black;
        }
        /* Changer la couleur du texte de la barre latérale en noir */
        [data-testid="stSidebar"] * {
            color: black;
        }
        </style>
        <div class="white-bar"></div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.image("/Users/maiav/Downloads/Openvalue Logo Rouge.png", use_column_width=True)
    st.header("Everything about your medical report :hospital:")
    pdf = st.file_uploader("Upload your PDF Files and Click on the Process Button", type="pdf")
    st.sidebar.title("Name(s) to ID(s) dictionnary :")

    if st.button("Process"):
        with st.spinner("Processing"):
            st.markdown(
            """
            <div style="background-color: white;
            height: 5px; /* Adjust height as needed */
            margin-top: 10px;">
            </div>
            """,
            unsafe_allow_html=True
            )
            if pdf is not None:
                extracted_text = extract_text_from_pdf(pdf)
                pdf_name = pdf.name
                extracted_text = re.sub(r'^\s*\d+\s*$', '', extracted_text, flags=re.MULTILINE)
                
                #Traduction de la langue du texte en Anglais 
                chunks = text_to_chunks(extracted_text)
                print("\nPremier appel\n")
                langue = response_to_question("Dis moi uniquement la langue du texte sans faire de phrase")
                print("\nFin Premier appel\n")

                if "Anglais" not in langue:
                    print("\nDeuxième appel\n")
                    texte_traduit = response_to_question(f"""
                        Translate ONLY this text word for word in English: 
                        {extracted_text} 
                        Give me only the translated this without saying anything else.
                    """)

                    print("\nFin Deuxième appel\n")
                    extracted_text = texte_traduit
                #extracted_text = extracted_text.replace("\n", " ")
                extracted_text = preprocess_text(extracted_text)
                extracted_text = " ".join(extracted_text)
                extracted_text = unidecode(extracted_text)

                print(extracted_text)

                cleaned_text = extracted_text

                #Modèle pour reconnaitre les noms des patients uniquement
                tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
                model = AutoModelForTokenClassification.from_pretrained("obi/deid_roberta_i2b2")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple",ignore_labels=[])

                max_chunk_size = 65

                chunks = []
                current_chunk = ""

                for word in extracted_text.split():
                    if len(current_chunk) + len(word) + 1 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += " " + word
                        else:
                            current_chunk = word
                    else:
                        chunks.append(current_chunk)
                        current_chunk = word

                if current_chunk:
                    chunks.append(current_chunk.strip())

                results = []
                for chunk in chunks:
                    chunk_results = nlp(chunk)
                    results.extend(chunk_results)

                for entity in results:
                    print(entity)
                    if entity['entity_group'] == 'AGE':
                        entity['word'] = entity['word'].strip()
                        cleaned_text = cleaned_text.replace(entity['word'],' X')
                    patients = [result['word'] for result in results if result['entity_group'] == 'PATIENT']
                    patients = [patient.strip() for patient in patients]

                name_to_id = {}
                index = []

                for result in results:
                    if result['entity_group'] == 'PATIENT':
                        if len(index)==0 and str(result['word']) not in str(name_to_id.values()):
                            index.append(result['end'])
                            current_person_id = 'ID{}'.format(len(name_to_id) + 1)
                            name_to_id[current_person_id] = result['word']
                        elif len(index)!=0:
                            if result['start'] == int(index[-1])+1 or result['start'] == int(index[-1])+2:
                                index.append(result['end'])
                                name_to_id[current_person_id] += result['word']
                                for a in results:
                                    if a['start'] == int(result['end']) and a['entity_group'] != 'PATIENT':
                                        index = []

                for cle in name_to_id:
                    name_to_id[cle] = name_to_id[cle].strip()

                for person in patients:
                    for cle in name_to_id:
                        if person in str(name_to_id[cle]):
                            cleaned_text = cleaned_text.replace(person,cle)


                #Modèle pour reconnaitre les dates de naissances
                print("\nTroisième appel\n")
                date_de_naissance = response_to_question("Tell me if the date of birth of the patient appears in th text below ?\n"+extracted_text+ "\nAnswer only by 'Yes' or 'No' : ")    
                print("\nFin Troisième appel\n")
                if 'Yes' in date_de_naissance:

                    tokenizer = AutoTokenizer.from_pretrained("tner/roberta-large-ontonotes5")
                    model = AutoModelForTokenClassification.from_pretrained("tner/roberta-large-ontonotes5")

                    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", ignore_labels=[])
                    ner_results = nlp(extracted_text)

                    dates = []
                    date_dic = {}

                    year_pattern = re.compile(r'\b\d{4}\b')

                    for result in ner_results:
                        if result['entity_group'] == 'DATE':
                            dates.append(result['word'])
                            match = year_pattern.search(result['word'])
                            if match:
                                date_dic[int(match.group())] = result['word']

                    cleaned_text = cleaned_text.replace(date_dic[min(date_dic)]," DATE_DE_NAISSANCE")


                #Modèle pour reconnaitre la ville
                print("\nQuatrième appel\n")
                ville = response_to_question(f"""
                    Tell me the city where the patient come from (city where he is born or where he lives) in the text below : {extracted_text} 
                    I want you to answer only with the exact name of the city as it appears in the text. Without the postal code.
                """)
                print("\nFin Quatrième appel\n")
                tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
                model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", ignore_labels=[])
                ner_results = nlp(extracted_text)

                for result in ner_results:
                    if result["entity_group"] == "LOC":
                        word_cleaned = result['word'].strip().lower()
                        ville_cleaned = str(ville).strip().lower()
                        if word_cleaned == ville_cleaned:
                            cleaned_text = cleaned_text.replace(result['word'],"VILLE")


                #Modèle pour reconnaitre les problèmes / traitements
                config = BertConfig.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

                tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
                model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner", config=config)

                nlp = pipeline("ner", model=model, tokenizer=tokenizer)
                ner_results = nlp(extracted_text)
                problem = []
                compteur = -1
                for result in ner_results:
                    if (result['entity'] == 'B-problem'):
                        print(result['entity'])
                        print(result['word'])
                        compteur+=1    
                        problem.append(result['word'])
                    if (result['entity'] == 'I-problem') and compteur != -1:
                        print(result['entity'])
                        print(result['word'])
                        problem[compteur] = problem[compteur] + ' '+result['word']
                    
                problems = [word.replace('#', '') for word in problem]

                treatment = []
                compteur = -1
                for result in ner_results:
                    if (result['entity'] == 'B-treatment'):
                        print(result['entity'])
                        print(result['word'])
                        compteur+=1    
                        treatment.append(result['word'])
                    if (result['entity'] == 'I-treatment') and compteur != -1:
                        print(result['entity'])
                        print(result['word'])
                        treatment[compteur] = treatment[compteur] +result['word']

                treatments = [word.replace('#', '') for word in treatment]

                #Modèle pour reconnaitre le numéro de téléphone
                print("\nCinquième appel\n")
                numéro = response_to_question(f"""
                    Tell me the phone number of the patient in the text below ?
                    {extracted_text}
                    I want you to answer only with the exact phone number as it appears in the text. If there is not the phone number of the patient answer 'None'.
                """)
                print("\nFin Cinquième appel\n")
                model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

                labels = ["phone number"]
                entities = model.predict_entities(extracted_text, labels)

                for entity in entities:
                    phone_number = entity["text"]
                    if numéro.strip()==phone_number.strip():
                        cleaned_text = cleaned_text.replace(phone_number,"NUMERO_DE_TELEPHONE")


                #Modèle pour reconnaitre les codes postaux
                print("\nSixième appel\n")
                code_postal = response_to_question("Tell me the postal code of the patient in the text below ?\n"+extracted_text+ "\nI want you to answer only with the exact postal code as it appears in the text don't do a sentence I just want the postal code. If there is not the postal code of the patient answer 'None'.")
                print("\nFin Sixième appel\n")
                tokenizer = AutoTokenizer.from_pretrained("zmilczarek/pii-detection-roberta-v2")
                model = AutoModelForTokenClassification.from_pretrained("zmilczarek/pii-detection-roberta-v2")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                ner_results = nlp(extracted_text)
                postal_code = []
                for result in ner_results:
                    if result["entity_group"] == "ID_NUM":
                        postal_code.append(result["word"])

                postal_code = [s.replace(" ", "") for s in postal_code]
                compteur = 0
                cleaned_postal_codes = []
                code = []
                for number in postal_code:
                    compteur += len(number)
                    code+=number
                    if compteur==5:
                        code = "".join(code)
                        cleaned_postal_codes.append(code)
                        compteur=0
                        code=[]
                for s in cleaned_postal_codes:
                    if s.strip() == code_postal.strip():
                        cleaned_text = cleaned_text.replace(s,"CODE_POSTAL")

                #Modèle pour reconnaitre l'adresse
                tokenizer = AutoTokenizer.from_pretrained("lakshyakh93/deberta_finetuned_pii")
                model = AutoModelForTokenClassification.from_pretrained("lakshyakh93/deberta_finetuned_pii")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                ner_results = nlp(extracted_text)

                for result in ner_results:
                    if result['entity_group'] == 'STREETADDRESS':
                        print(result['word'])
                        print("\nSeptième appel\n")
                        adresse = response_to_question(f"""
                            Tell me only if {result['word']} is a part of the adress of the patient in this text :{extracted_text}.
                            Only answer by 'Yes' or 'No'. Nothing more.
                        """)
                        print("\nFin Septième appel\n")
                        if 'Yes' in adresse:
                            cleaned_text = cleaned_text.replace(result['word']," ADRESSE")


                #Utilisation d'un LLM pour affiner le travail du NER
                chunks = text_to_chunks(extracted_text)
                print("\nHuitieme appel\n")
                name_to_id = response_to_question(f"""
                Here is a dictionary of names/IDs I have created: {name_to_id}
                - It indicates an ID for each patient's name present in the text.
                From this text: {extracted_text}
                - Correct the dictionary if there are any errors; I do not want names of places/buildings, addresses, or doctors. If such names appear, remove them from the dictionary.
                - Only provide the corrected dictionary without any intrducing sentence.
                """)

                print(name_to_id)
                print("\nFin Huitième appel\n")
                print("\nNeuvième appel\n")
                problems = response_to_question(f"""
                    Here is a list of problems/diseases I found : {str(problems)}
                    -From this text : {extracted_text}
                    -Correct the list of problems I gave you only with real problems without forgeting any and without repetition. 
                    -I only want the list stored in a python list. 
                    -I only want the corrected list. 
                    -Answer me with only the better list.
                    -If there is no problems just answer : 'None'. 
                    Give me only the list without sentences like 'Here is ...'
                """)
                print(problems)
                print("\nFin Neuvième appel\n")
                print("\nDixième appel\n")
                treatments = response_to_question(f"""
                    Here is a list of treatments I recognized in the text : {str(treatments)}
                    From this text :{extracted_text}
                    -Correct the list of treatments I gave you only with real treatments without forgeting any and make sure there is no repetition. 
                    -Only keep treatments that are medical related.
                    -If there is no treatments just answer : 'None'. 
                    -Answer me with only the corrected list you recognized in the text not with the one i gave you.
                    -Give directly the list
                    -Don't add any sentence
                """)
                print(treatments)
                print("\nFin Dixième appel\n")
                print("\nOnzième appel\n")
                pseudonymized_text = response_to_question(f"""
                    From this text: 
                    {cleaned_text}
                    - Replace the patient's date of birth with 'DATE_DE_NAISSANCE' don't replace any other date.
                    - Replace {ville} with 'VILLE' where it refers to the city where the patient lives or was born. Leave 'VILLE' if already correctly replaced. Replace only if the city is linked to the patient
                    - Replace {code_postal} with 'CODE_POSTAL'. Only replace this specific zip code.
                    - Replace the patient's phone number with 'NUMERO_DE_TELEPHONE'.
                    - Ensure the patient's address (street name and number) is replaced by 'ADRESSE'.
                    - Replace the patient's age with 'X', usually near 'year old'.
                    - Keep 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL', 'ID', 'ADRESSE', 'NUMERO_DE_TELEPHONE', and 'X' as they are if they appear in the pseudonymized text i gave you.
                    - If none of these changes apply, return the text as is.
                    -Make sure the ID of the patient are not replaced.
                    Only answer with one final pseudonymized text.
                """)
                print("\nOnzième appel\n")
                pseudonymized_text = response_to_question(f"""
                    Based on this text : 
                    {extracted_text}
                    Here is the pseudonymized text I created : 
                    {cleaned_text}
                    With the first text and the following table :  
                    {name_to_id}
                    -Ensure that each name present in the table has been correctly replaced by its ID in the pseudonymized text. 
                    -In the text you generate, you must only pseudonymize the person who is in the table. 
                    -VERY IMPORTANT: Make sure the name or first name of the person in the table does not appear in the text without being replaced by their ID. 
                    -If a person who is present in the table is mentioned in the text, replace their entire name/first name in the text with their ID. 
                    -If I have already pseudonymized names/first names, just verify that it has been done correctly and LEAVE the IDs I have put in the text without any modification. 
                    -Double-check that the name present in the table is correctly replaced by its ID when it appears in the text and that their name does not appear in the text anymore; this is crucial. 
                    -VERY IMPORTANT if the words 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL', 'NUMERO_DE_TELEPHONE', or 'X' (the age of the patient) are present in the pseudonymized text I gave you, make sure to incorporate them as they are and don't modify them it's very important that they stay as they are. 
                    -Give me only the well-pseudonymized text without any introduction sentences like 'Here is ...'
                """)
                
                print("\nFin Onzième appel\n")
                print(pseudonymized_text)
                print("\nDouzième appel\n")
                pseudonymized_text = response_to_question(f"""
                    From this text: 
                    {pseudonymized_text}
                    - Replace the patient's date of birth with 'DATE_DE_NAISSANCE' don't replace any other date.
                    - Replace {ville} with 'VILLE' where it refers to the city where the patient lives or was born. Leave 'VILLE' if already correctly replaced. Replace only if the city is linked to the patient
                    - Replace {code_postal} with 'CODE_POSTAL'. Only replace this specific zip code.
                    - Replace the patient's phone number with 'NUMERO_DE_TELEPHONE'.
                    - Ensure the patient's address (street name and number) is replaced by 'ADRESSE'.
                    - Replace the patient's age with 'X', usually near 'year old'.
                    - Keep 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL', 'ID', 'ADRESSE', 'NUMERO_DE_TELEPHONE', and 'X' as they are if they appear in the pseudonymized text i gave you.
                    - If none of these changes apply, return the text as is.
                    -Make sure the ID of the patient are not replaced.
                    Only answer with one final pseudonymized text.
                """)
                print("\nFin Douzième appel\n")
                print("Name(s) / ID(s) dictionary :\n"+name_to_id+"\n\n Problem(s) / Disease(s) :\n"+problems+"\n\n Treatment(s) :\n"+treatments+"\n\n Pseudonymised text : \n"+pseudonymized_text)
                #Affichage des informations sur streamlit
                name_to_id = ast.literal_eval(name_to_id.strip().replace('\n', ''))
                for clé, valeur in name_to_id.items():
                    st.sidebar.text(str(clé) + " " + valeur)

                with st.container(border=True):  
                    st.subheader("Pseudonymized text :")
                    st.write(pseudonymized_text)

                with st.container(border=True):
                    st.subheader("Problem(s) / Disease(s) :")
                    if problems == "Aucun":
                        st.text(problems)
                    else:
                        problems = string_to_list(problems)
                        for problem in problems:
                            st.text(problem)

                with st.container(border=True):            
                    st.subheader("Treatment(s) :")
                    if treatments == "Aucun":
                        st.text(treatments)
                    else:
                        treatments = string_to_list(treatments)
                        for treatment in treatments:
                            st.text(treatment)

                #Creation du fichier contenant le rapport médical pseudonymisé
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, pseudonymized_text)
                repertoire = "." 
                st.subheader("Output :")

                if os.access(repertoire, os.W_OK):
                    os.makedirs(repertoire, exist_ok=True)
                    chemin_fichier = os.path.join(repertoire, "pseudonymised"+pdf_name)
                    pdf.output(chemin_fichier)
                    repertoire_courant = os.getcwd()
                    st.write("Le rapport médical pseudonymisé a été créé et sauvegardé sous le nom : "+repertoire_courant+"/pseudonymised_"+pdf_name)
                else:
                    st.write("Vous n'avez pas les droits pour créer un fichier dans ce répertoire.")

if __name__ == "__main__":
    main() 