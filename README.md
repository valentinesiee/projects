# Projects

Welcome to my project showcase! Below are some of the projects I developed primarily during my internship.

## ChatURL 
ChatURL is a StreamLit application designed for summarizing any website's content. Simply input the URL, and it provides a concise summary using RAG (Retrieve, Aggregate, Generate) methodology coupled with a LLM.

## ChatPDF 
ChatPDF is another StreamLit application tailored for summarizing PDF documents. Upload your PDF file, and receive a summary using RAG methodology, powered by a LLM.

## Medical Reports Pseudonymization 
One of the main projects I undertook during my internship was the pseudonymization of medical reports. The aim was to safeguard sensitive patient information like names and dates of birth typically included in doctors' reports. The objective was to assign a unique ID to each patient and replace their name occurrences in the text with this ID. A significant challenge was ensuring that this process did not inadvertently alter names of locations, such as hospitals , addresses or names of non-patient persons like doctors. Once the dictionary associating IDs with patient names was established, it would be exclusively utilized by the hospital to match patients with their reports.
Additionally, the project aimed to conceal dates of birth while ensuring other date references remained unaffected.
All this was achieved through a combination of Named Entity Recognition (NER) and LLM techniques.
