# Projects

Welcome to my project showcase! Below are some of the projects I developed during my internships and personal work.

---

## ChatURL

**Description:**
ChatURL is a Streamlit application designed to summarize the content of any website. Simply input the URL, and it generates a concise summary using the RAG (Retrieve, Aggregate, Generate) methodology powered by a LLM.

**Key steps / features:**

* Web content extraction (scraping and text cleaning).
* Text chunking and vectorization for retrieval.
* RAG pipeline: retrieve relevant passages → aggregate → generate summary.
* Streamlit interface for input/output.

**Tech stack:** Python, Streamlit, BeautifulSoup / newspaper3k, FAISS / Annoy, HuggingFace / OpenAI, LangChain (optional).

---

## ChatPDF

**Description:**
ChatPDF is a Streamlit application for summarizing PDF documents. Upload a PDF file and receive a summary produced through a RAG pipeline combined with a LLM.

**Key steps / features:**

* PDF text extraction (multi-page handling, OCR if needed).
* Chunking and embedding for retrieval.
* RAG + LLM pipeline for coherent summarization.
* Simple interface for file upload and summary display.

**Tech stack:** Python, Streamlit, PyPDF2 / PDFMiner, Tesseract (OCR), FAISS, HuggingFace / OpenAI, LangChain.

---

## Medical Reports Pseudonymization

**Description:**
One of my main internship projects: pseudonymization of medical reports to protect sensitive patient information (names, dates of birth, etc.) while keeping the reports useful for analysis.

**Challenges & goals:**

* Replace patient names with unique IDs without altering hospital names, doctor names, or addresses.
* Conceal dates of birth while preserving other temporal references.
* Extract patient diseases and treatments for structured insights.

**Approach:**

* Named Entity Recognition (NER) models to detect sensitive entities.
* Post-processing heuristics to differentiate people vs. locations/institutions.
* LLM-based verification to reduce false positives/negatives.
* Secure ID ↔ patient dictionary, accessible only by hospitals.

**Tech stack:** Python, spaCy / HuggingFace Transformers, LLMs, secure storage solutions.

---

## Bluetooth Attacks — Data Exploration & Modeling

**Description:**
A complete end-to-end data science project analyzing Bluetooth traffic/attacks. It covered EDA, preprocessing, model selection, and evaluation to build a predictive pipeline for anomaly detection or classification.

**Key steps:**

1. Exploratory Data Analysis (EDA): descriptive stats, correlations, anomaly detection.
2. Data preprocessing: cleaning, imputation, normalization, encoding.
3. Feature engineering: time-based features, window aggregation, signal features.
4. Model selection.
5. Validation.
6. Interpretability.

**Results:**

* Built a robust classification pipeline with strong performance.
* Delivered insights on most important predictive features.
* Recommendations for potential production deployment (monitoring & alerting).

**Tech stack:** Python, Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn.

---

## CNN From Scratch

**Description:**
A pedagogical project where I implemented a Convolutional Neural Network (CNN) **from scratch** (without high-level DL libraries) to deeply understand convolution layers, backpropagation, and model training.

**Key steps:**

1. Dataset: small-scale image dataset.
2. Implementation of layers: convolution, ReLU, pooling, flattening, fully connected, softmax.
3. Forward & backward pass: manual gradient computation and parameter updates (SGD, momentum).
4. Training loop: batching, cross-entropy loss, monitoring accuracy.
5. Improvements: regularization, data augmentation, early stopping.
6. Evaluation: learning curves, confusion matrix, visualization of learned filters.

**Learnings & results:**

* Built a fully functional CNN with only NumPy.
* Gained hands-on understanding of backpropagation in conv layers.
* Compared performance with PyTorch/TensorFlow implementations.

**Tech stack:** Python, NumPy, Matplotlib, optional PyTorch/TensorFlow for benchmarking.

