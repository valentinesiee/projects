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



def pdf_to_text(pdfs):
    """
    Extrait le contenu d'un pdf

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
    Divise le texte en chunks

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
    Calcule le vecteur associé a chaque chunk et les stocke dans une base de données

    chunks (liste) : Le PDF en chunks

    Returns:
        liste: Les vecteurs
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectordb = FAISS.from_texts(chunks, embeddings)
    return vectordb

def response_to_question(question,vectordb):
    """
    Réponse obtenue en posant une question au LLM

    question (string) : La question que l'on pose au LLM
    vectordb(list) : Base de connaisance qui va permettre au LLM de répondre

    Returns:
        string: La réponse à la question
    """
    docs = vectordb.similarity_search(question)
    llm = ChatCohere(cohere_api_key="3We4XANs6kzV7TJD3dagFI02ul4eS25eNxl0tV24")
    chain = load_qa_chain(llm,chain_type="stuff")
    answer = chain.run(input_documents=docs,question=question)
    return answer

def extract_text_from_pdf(pdf_path):
    """
    Extrait le contenu d'un pdf

    pdf_path (string) : Chemin relatif ou absolu menant au PDF

    Returns:
        string: Le contenu textuel du PDF
    """
    with fitz.open(pdf_path) as pdf_document:
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            page_text = page.get_text()
            text += page_text
    return text

def string_to_list(problems_str):
    """
    Crée une liste à partir d'une string

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
                #pdf_path = 'medicalreport3fr.pdf' 
                extracted_text = extract_text_from_pdf(pdf)
                pdf_name = pdf.name

                #Traduction de la langue du texte en Anglais 
                chunks = text_to_chunks(extracted_text)
                vectordb = chunks_to_vectors(chunks)
                langue = response_to_question("Dis moi uniquement la langue du texte sans faire de phrase",vectordb)

                if langue != "Anglais.":
                    texte_traduit = response_to_question("Traduis moi UNIQUEMENT ce texte mot pour mot en anglais sans RIEN ajouter en plus",vectordb)
                    extracted_text = texte_traduit
                print("\n\nTexte traduit : \n"+extracted_text)

                #Premier modèle pour reconnaitre les noms des patients uniquement
                tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
                model = AutoModelForTokenClassification.from_pretrained("obi/deid_roberta_i2b2")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                ner_results = nlp(extracted_text)

                patients = [result['word'] for result in ner_results if result['entity_group'] == 'PATIENT']
                patients = [patient.strip() for patient in patients]
                
                name_to_id = {}
                index = []
                cleaned_text = extracted_text
                for result in ner_results:
                    if result['entity_group'] == 'PATIENT':
                        if len(index)==0 and str(result['word']) not in str(name_to_id.values()):
                            index.append(result['end'])
                            current_person_id = 'ID{}'.format(len(name_to_id) + 1)
                            name_to_id[current_person_id] = result['word']
                        elif len(index)!=0:
                            if result['start'] == int(index[-1])+1:
                                index.append(result['end'])
                                name_to_id[current_person_id] += result['word']
                                index = []

                for cle in name_to_id:
                    name_to_id[cle] = name_to_id[cle].strip()

                for person in patients:
                    for cle in name_to_id:
                        if person in str(name_to_id[cle]):
                            cleaned_text = cleaned_text.replace(person,cle)

                #Deuxième modèle pour reconnaitre les dates de naissances
                date_de_naissance = response_to_question("Dis moi uniquement si oui ou non il y a la date de naissance du patient présent dans le texte sans faire de phrase",vectordb)    

                if date_de_naissance:

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
                            
                #Deuxième modèle nécéssaire pour créer le dictionnaire
                
                # config = BertConfig.from_pretrained("dslim/bert-base-NER")

                # config.hidden_dropout_prob = 0.2
                # config.attention_probs_dropout_prob = 0.2
                # config.num_hidden_layers = 12

                # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", config=config)
                # tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

                # nlp = pipeline("ner", model=model, tokenizer=tokenizer, ignore_labels=[])
                # ner_results = nlp(cleaned_text)

                # #Création de la table pour mapper chaque nom a un ID
                # name_to_id = {}
                # current_person_id = None
                # cleaned_text = []
                # noms = []
                
                # for result in ner_results:
                #     if result['entity'] == 'B-PER':
                #         if result['word'] in patients:
                #             person_name = result['word']
                #             if person_name not in noms:
                #                 noms.append(person_name)
                #                 current_person_id = 'ID{}'.format(len(name_to_id) + 1)
                #                 name_to_id[current_person_id] = person_name
                #                 cleaned_text.append(current_person_id)
                #             else:
                #                 for cle, valeur in name_to_id.items():
                #                     if person_name in valeur:
                #                         cleaned_text.append(cle)
                #         else:
                #             for verif in ner_results:
                #                 if int(verif['index'])==result['index']+1:
                #                     if "#" in verif['word']:
                #                         person = result['word']+verif['word'].replace('##','')
                #                         if person in noms:
                #                             for cle, valeur in name_to_id.items():
                #                                 if person in valeur:
                #                                     cleaned_text.append(cle)
                #                     else:
                #                         cleaned_text.append(result['word'])
                                    
                    
                #     elif result['entity'] == 'I-PER':
                #         if result['word'] in patients:
                #             noms.append(result['word'])
                #             person_name += ' ' + result['word']  
                #             name_to_id[current_person_id] = person_name.replace('##', '')
                #             cleaned_text.append(current_person_id)
                            
                #         if "#" in result['word']:
                #             for a in ner_results:
                #                 if a['index'] == int(result['index'])-1:
                #                     b = a['word'] + result['word'].replace('##','')
                #                     if b in patients:
                #                         temp = []
                #                         for cle, valeur in name_to_id.items():
                #                             temp.append(valeur)
                #                         if b not in valeur:
                #                             noms.append(b)
                #                             person_name = person_name+" "+b
                #                             name_to_id[current_person_id] = person_name
                #                             cleaned_text.append(current_person_id)
                #                         else:
                #                             cleaned_text.append(current_person_id)

                #                     else:
                #                         cleaned_text.append(b)
                #     else:
                #         cleaned_text.append(result['word'])

                # cleaned_text = ' '.join(cleaned_text)

                # cleaned_text = re.sub(r'\s*##\s*', '', cleaned_text)
                

                #Quatrième modèle pour reconnaitre l'adresse
                ville = response_to_question("Dis moi uniquement d'où vient le patient sans faire de phrase. Ecris moi la ville exactement comme elle apparait dans le texte. S'il y en a pas dis moi juste : Aucun",vectordb)
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

                #Cinquième modèle pour reconnaitre les problèmes / traitements
                config = BertConfig.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

                tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")
                model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner", config=config)

                nlp = pipeline("ner", model=model, tokenizer=tokenizer)
                ner_results = nlp(extracted_text)
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

                #Sixième modèle pour reconnaitre le numéro de téléphone
                numéro = response_to_question("Dis moi uniquement le numéro de téléphone du patient sans faire de phrase. Ecris moi le numéro exactement comme il apparait dans le texte. S'il y en a pas dis moi juste : Aucun",vectordb)
    
                model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

                labels = ["phone number"]
                entities = model.predict_entities(extracted_text, labels)

                for entity in entities:
                    phone_number = entity["text"]
                    if numéro==phone_number:
                        cleaned_text = cleaned_text.replace(phone_number,"NUMERO_DE_TELEPHONE")

                #Septième modèle pour reconnaitre les codes postaux
                code_postal = response_to_question("Dis moi uniquement le code postal du patient sans faire de phrase. Il s'agit soit du code postal associer a la ville où il est née ou à la ville où il habite.  Ecris moi le code postal exactement comme il apparait dans le texte. S'il y en a pas dis moi juste : Aucun",vectordb)
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
                    if s == code_postal:
                        cleaned_text = cleaned_text.replace(s,"CODE_POSTAL")

                #Huitième modèle pour reconnaitre l'adresse
                tokenizer = AutoTokenizer.from_pretrained("lakshyakh93/deberta_finetuned_pii")
                model = AutoModelForTokenClassification.from_pretrained("lakshyakh93/deberta_finetuned_pii")

                nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
                ner_results = nlp(extracted_text)

                for result in ner_results:
                    if result['entity_group'] == 'STREETADDRESS':
                        adresse = response_to_question("Dis moi uniquement si "+result['word']+" fait partis de l'adresse du patient. Reponds uniquement par 'Oui' ou 'Non'. " ,vectordb)
                        if 'Oui' in adresse:
                            cleaned_text = cleaned_text.replace(result['word']," ADRESSE")

                #Utilisation d'un LLM pour affiner le travail du NER
                print(cleaned_text)
                chunks = text_to_chunks(extracted_text)
                vectordb = chunks_to_vectors(chunks)
                name_to_id = response_to_question("Voici un dictionnaire que j'ai établie en indiquant un ID pour chaque Nom/prénom de patient présent dans le texte : \n"+str(name_to_id)+" A partir du texte corrige le dictionnaire s'il y a des erreurs en effet je ne veux pas de noms de lieux/batiments ou encore d'adresse ou de medecins. Si cela arrive supprime cette si-disante personne du dictionnaire. Je veux uniquement le dictionnaire.",vectordb)
                problems = response_to_question("Voici une liste de problèmes/maladies que j'ai remarqué : "+str(problems)+"\nA partir du texte corrige la liste de problemes que je viens de donner en n'en oubliant aucun et en ne faisant pas de répétition. Je veux uniquement la liste stockée dans une liste python. S'il n'y en a pas indique juste : Aucun",vectordb)
                treatments = response_to_question("Voici une liste de traitements/soins que j'ai remarqué : "+str(treatments)+"\nA partir du texte corrige la liste de traitements que je viens de donner en n'en oubliant aucun et en indiquant les doses si elles sont indiquées. Je veux uniquement la liste stockée dans une liste python. S'il n'y en a pas indique juste : Aucun",vectordb)
                #pseudonymized_text = response_to_question("Voici le texte pseudonymisé que j'ai fais : \n"+cleaned_text+"\n\n Cependant à partir du texte original ainsi qu'avec la table suivante : \n"+name_to_id+"\n\nAssure toi que j'ai correctement remplacé les noms par les IDs correspondant se trouvant uniquement dans la table que je viens de te donner, donne beaucoup d'importance à cette table, si un nom n'apparaissant pas dans ta table a été remplacé alors laisse le tel qu'il est dans le texte original.\nDe plus si un ID apparait dans le texte alors qu'il n'est pas dans la table remplace le par le nom de base.\nIl ne faut surtout pas que tu pseudonymises avec un ID qui n'est pas dans la table.\nNE CREE SURTOUT PAS DE NOUVEL ID.\nDans le texte que tu vas générer il faut vraiment que tu pseudonymises uniquement la personne qui est dans la table. TRES IMPORTANT il ne faut surtout pas que le nom de la personne qui est dans la table apparaisse dans le texte sans être remplacé par son ID il faut donc que tu t'assures qu'uniquement l'ID doit etre présent et PAS le nom/prénom de cette personne. Si j'ai remplacé un nom par un ID vérifie uniquement que c'est bien le bon ID associé au nom dans la table si c'est le cas laisse tel que j'ai remplacé. Réponds sans faire de phrase et surtout si 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL' ou 'NUMERO_DE_TELEPHONE' sont présents dans le texte que je t'ai donné, laisse-les tel quel en toute lettre.",vectordb)
                pseudonymized_text = response_to_question("Voici le texte pseudonymisé que j'ai fais : \n"+cleaned_text+"\n\n De plus avec la table suivante : \n"+name_to_id+"\n\nAssure toi que chaque Nom présent dans la table à correctement été remplacé par son ID, il faut uniquement que tu suives cette table.\nIl ne faut surtout pas que tu remplaces avec un ID qui n'est pas dans la table.\nNE CREE SURTOUT PAS DE NOUVEL ID.\nDans le texte que tu vas générer il faut vraiment que tu pseudonymises uniquement la personne qui est dans la table. Son Nom/Prénom ne doit donc plus apparaitre dans le texte tu vas me donner. TRES IMPORTANT il ne faut surtout pas que le nom ou prénom de la personne qui est dans la table apparaisse dans le texte sans être remplacé par son ID. En effet lorsqu’une personne qui est présente dans la table est citée dans le texte il faut remplacer l'entiereté de son nom/prénom dans le texte par son ID. Si j'ai déja pseudonymisé des Noms/prénoms vérifie juste que cela a bien été fait et LAISSE les IDs que j'ai mis dans le texte sans rien modifier. Si tu vois un ID1 dans le texte que je t'ai donné assure toi uniquement qu'il a correctement remplacé la bonne personne. Donc si tu vois un ID dans le texte que je t'ai donné assure toi juste que cela a bien été fait. Vérifie bien une derniere fois que le nom présent dans la table est bien remplacé par son ID quand il apparait dans le texte et que son nom n'apparaisse plus dans le texte ceci est primordial. Réponds sans faire de phrase et surtout si 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL' ou 'NUMERO_DE_TELEPHONE' sont présents dans le texte que je t'ai donné, laisse-les tel quel en toute lettre.",vectordb)
                pseudonymized_text = response_to_question("A partir de ce texte : \n\n"+pseudonymized_text+"\n- La date de naissance du patient doit être remplacée par 'DATE_DE_NAISSANCE'. Si cela n'a pas été fait, fais-le. \n- Le nom de la ville où habite le patient doit être remplacé par 'VILLE'. Si cela n'a pas été fait, remplace "+ville+" par 'VILLE'. Si 'VILLE' apparaît déjà, cela signifie que cela a été fait correctement, dans ce cas laisse tel quel. Remplace également le nom de la ville où est née le patient par 'VILLE'.\n- Le code postal de la ville associée au patient (5 chiffres) doit être remplacé par 'CODE_POSTAL'. Si cela n'a pas été fait, remplace "+code_postal+" par 'CODE_POSTAL'. Il faut uniquement que tu remplaces ce code postal : "+code_postal+".\n- Le numéro de téléphone du patient doit être remplacé par 'NUMERO_DE_TELEPHONE'. Si cela n'a pas été fait, fais-le.\nAssure-toi également que les noms des personnes qui n'ont pas été pseudonymisés ne sont pas coupés en plein milieu. \nJe veux uniquement que tu me donnes le texte bien pseudonymisé en français en traduisant mot pour mot. Important : si 'DATE_DE_NAISSANCE', 'VILLE', 'CODE_POSTAL', 'ID' ou 'NUMERO_DE_TELEPHONE' sont présents dans le texte, laisse-les tels quels.", vectordb)

                print("Name(s) / ID(s) dictionary :\n"+name_to_id+"\n\n Problem(s) / Disease(s) :\n"+problems+"\n\n Treatment(s) :\n"+treatments+"\n\n Pseudonymised text : \n"+pseudonymized_text)

                #Affichage des informations sur streamlit
                name_to_id = ast.literal_eval(name_to_id)
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

                #Creation du rapport médical pseudonymisé
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