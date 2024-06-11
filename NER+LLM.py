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

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf_document:
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            page_text = page.get_text()
            text += page_text
    return text

def string_to_list(problems_str):
    cleaned_str = problems_str.strip()
    try:
        problems_list = ast.literal_eval(cleaned_str)
        return problems_list
    except (ValueError, SyntaxError):
        st.error("The input string is not in the correct format.")
        return []
    
def main(): 
    st.set_page_config("Medical report summary")

    st.markdown(
        f"""
        <style>
        /* Applique la couleur de fond blanche à la barre latérale */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative; /* Permet de positionner le logo par rapport à la barre latérale */
        }}
        /* Applique la couleur de fond rouge à l'application principale */
        .stApp {{
            background-color: #E35656;
        }}

        /* Rendre le titre noir */
        .css-10trblm {{  /* Remplacer par la classe appropriée pour le titre, si nécessaire */
            color: black;
        }}
        /* Changer la couleur du texte de la barre latérale en noir */
        [data-testid="stSidebar"] * {{
            color: black;
        }}
        """,
        unsafe_allow_html=True
    )
    st.sidebar.image("/Users/maiav/Downloads/Openvalue Logo Rouge.png", use_column_width=True)
    st.header("Everything about your medical report :hospital:")
    pdf = st.file_uploader("Upload your PDF Files and Click on the Process Button", type="pdf")
    st.sidebar.title("Name(s) to ID(s) dictionnary")

    if pdf is not None:
        #pdf_path = 'medicalreport3fr.pdf' 
        extracted_text = extract_text_from_pdf(pdf)
        pdf_name = pdf.name

        #Traduction de la langue du texte en Anglais 
        chunks = text_to_chunks(extracted_text)
        vectordb = chunks_to_vectors(chunks)
        langue = response_to_question("Dis moi uniquement la langue du texte sans faire de phrase",vectordb)

        if langue != "Anglais":
            texte_traduit = response_to_question("Traduis moi uniquement le texte en anglais",vectordb)
            extracted_text = texte_traduit

        #Premier modèle pour reconnaitre les noms des patients uniquement
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
        model = AutoModelForTokenClassification.from_pretrained("obi/deid_roberta_i2b2")

        from transformers import pipeline

        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        ner_results = nlp(extracted_text)

        patients = [result['word'] for result in ner_results if result['entity_group'] == 'PATIENT']
        patients = [patient.strip() for patient in patients]
        
        #Deuxième modèle pour reconnaitre les dates de naissances

        date_de_naissance = response_to_question("Dis moi uniquement si oui ou non il y a la date de naissance du patient présent dans le texte sans faire de phrase",vectordb)    

        if date_de_naissance:

            from transformers import AutoTokenizer, AutoModelForTokenClassification

            tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
            model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")

            from transformers import pipeline

            nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple", ignore_labels=[])
            ner_results = nlp(extracted_text)

            dates = []
            extracted_textDate = []
            date_dic = {}

            year_pattern = re.compile(r'\b\d{4}\b')

            for result in ner_results:
                if result['entity_group'] == 'DATE':
                    dates.append(result['word'])
                    match = year_pattern.search(result['word'])
                    if match:
                        date_dic[int(match.group())] = result['word']

            for result in ner_results:
                if result['word'] == date_dic[min(date_dic)]:
                    extracted_textDate.append("DATE DE NAISSANCE")
                else:
                    extracted_textDate.append(result['word'])
                    
            extracted_text = ' '.join(extracted_textDate)

        #Troisième modèle nécéssaire pour créer le dictionnaire
        config = BertConfig.from_pretrained("Babelscape/wikineural-multilingual-ner")

        config.hidden_dropout_prob = 0.2
        config.attention_probs_dropout_prob = 0.2
        config.num_hidden_layers = 12

        model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner", config=config)
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")

        nlp = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=[])
        ner_results = nlp(extracted_text)

        #Création de la table pour mapper chaque nom a un ID
        name_to_id = {}
        current_person_id = None
        pseudonymized_text = []
        noms = []

        for result in ner_results:
            if result['entity'] == 'B-PER' and result['word'] in patients:
                person_name = result['word']
                if person_name not in noms:
                    noms.append(person_name)
                    current_person_id = 'ID{}'.format(len(name_to_id) + 1)
                    name_to_id[current_person_id] = person_name
                    pseudonymized_text.append(current_person_id)
                else:
                    pseudonymized_text.append("he")
            elif result['entity'] == 'I-PER' and result['word'] in patients:
                noms.append(result['word'])
                person_name += ' ' + result['word']  
                name_to_id[current_person_id] = person_name.replace('##', '')
                pseudonymized_text.append(current_person_id)
            else:
                pseudonymized_text.append(result['word'])

        pseudonymized_text = ' '.join(pseudonymized_text)

        cleaned_text = re.sub(r'\s*##\s*', '', pseudonymized_text)

        #Quatrième modèle pour reconnaitre les problèmes / traitements
        config = BertConfig.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

        tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
        model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner", config=config)

        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        ner_results = nlp(cleaned_text)

        problem = []
        compteur = -1
        for result in ner_results:
            if (result['entity'] == 'B-problem'):
                #print(result['word'])
                compteur+=1    
                problem.append(result['word'])
            if (result['entity'] == 'I-problem'):
                #print(result['word'])
                problem[compteur] = problem[compteur] + ' '+result['word']
            
        problems = [word.replace('#', '') for word in problem]

        treatment = []
        compteur = -1
        for result in ner_results:
            if (result['entity'] == 'B-treatment'):
                compteur+=1    
                treatment.append(result['word'])
            if (result['entity'] == 'I-treatment'):
                treatment[compteur] = treatment[compteur] +result['word']

        treatments = [word.replace('#', '') for word in treatment]

        #Utilisation d'un LLM pour affiner le travail du NER
        chunks = text_to_chunks(extracted_text)
        vectordb = chunks_to_vectors(chunks)
        name_to_id = response_to_question("Voici un dictionnaire que j'ai établie en indiquant un ID pour chaque Nom/prénom de patient présent dans le texte : \n"+str(name_to_id)+" A partir du texte corrige le dictionnaire s'il y a des erreurs en effet je ne veux pas de noms de lieux/batiments ou encore d'adresse ou de medecins. Si cela arrive supprime cette si-disante personne du dictionnaire. Je veux uniquement le dictionnaire.",vectordb)
        problems = response_to_question("Voici une liste de problèmes/maladies que j'ai remarqué : "+str(problems)+"\nA partir du texte corrige la liste de problemes que je viens de donner en n'en oubliant aucun et en ne faisant pas de répétition. Je veux uniquement la liste stockée dans une liste python. S'il n'y en a pas indique juste : Aucun",vectordb)
        treatments = response_to_question("Voici une liste de traitements/soins que j'ai remarqué : "+str(treatments)+"\nA partir du texte corrige la liste de traitements que je viens de donner en n'en oubliant aucun et en indiquant les doses si elles sont indiquées. Je veux uniquement la liste stockée dans une liste python. S'il n'y en a pas indique juste : Aucun",vectordb)
        pseudonymized_text = response_to_question("Voici le texte synonymiser que j'ai fais : "+cleaned_text+"\nCependant à partir du texte original ainsi qu'avec la table suivante : \n"+name_to_id+" Assure toi que j'ai correctement remplacé les noms par les IDs correspondant se trouvant uniquement dans la table que je viens de te donner, donne plus d'importances à cette table, si un nom n'apparaissant pas dans ta table a été remplacé alors laisse le tel qu'il est dans le texte original. De plus si un ID apparait dans le texte alors qu'il n'est pas dans la table remplace le par le nom de base. J'ai également remplacé la date de naissance du patient par DATE DE NAISSANCE si elle était présente dans le texte original assure toi que cela à également bien été fait si ce n'est pas le cas fais le. Je veux uniquement que tu me donnes le texte bien synonymisé en français en traduisant mot pour mot.",vectordb)
        print("Name(s) / ID(s) dictionary :\n"+name_to_id+"\n\n Problem(s) / Disease(s) :\n"+problems+"\n\n Treatment(s) :\n"+treatments+"\n\n Pseudonymised text : \n"+pseudonymized_text)

        #Affichage des informations sur streamlit
        name_to_id = ast.literal_eval(name_to_id)
        for clé, valeur in name_to_id.items():
            st.sidebar.text(clé + " " + valeur)
        st.subheader("Pseudonymized text :")
        st.write(pseudonymized_text)
        st.subheader("Problem(s) / Disease(s) :")
        if problems == "Aucun":
            st.text(problems)
        else:
            problems = string_to_list(problems)
            for problem in problems:
                st.text(problem)
        st.subheader("Treatment(s) :")
        if treatments == "Aucun":
            st.text(treatments)
        else:
            treatments = string_to_list(treatments)
            for treatment in treatments:
                st.text(treatment)

        #Creation du rapport médical pseudonymiser
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