import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import difflib
import gensim.downloader as api

# Download NLTK resources if not already downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer and stopwords
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to tokenize and stem text
def tokenize_and_stem(text):
    word_tokens = word_tokenize(text.lower())
    stemmed_words = [porter.stem(word) for word in word_tokens if word not in stop_words]
    return stemmed_words

# Function to stem a sentence
def stem_sentence(sentence):
    token_words = word_tokenize(sentence.lower())
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

# Function to calculate kappa index
def kappa_index(mxI, myJ):
    mxI_classes = [round(value * 10) for value in mxI]
    myJ_classes = [round(value * 10) for value in myJ]

    score = cohen_kappa_score(mxI_classes, myJ_classes)
    index = ""
    if score <= 0.00:
        index = "Less than chance-agreement"
    elif score >= 0.01 and score <= 0.20:
        index = "Slight agreement"
    elif score >= 0.21 and score <= 0.40:
        index = "Fair agreement"
    elif score >= 0.41 and score <= 0.60:
        index = "Moderate agreement"
    elif score >= 0.61 and score <= 0.80:
        index = "Substantial agreement"
    elif score >= 0.81 and score <= 1.00:
        index = "Almost perfect agreement"
    else:
        index = "Index not defined"
    return score, index

# Function to pad a matrix
def pad_matrix(matrix, target_shape):
    padded_matrix = np.zeros(target_shape)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

# Function to get similar sentences using SequenceMatcher
def get_similar_sentences(sentences1, sentences2):
    similar_sentences = []
    for sentence1 in sentences1:
        max_similarity = 0
        similar_sentence = ""
        for sentence2 in sentences2:
            similarity = difflib.SequenceMatcher(None, sentence1, sentence2).ratio()
            if similarity >= max_similarity:
                max_similarity = similarity
                similar_sentence = sentence2
        similar_sentences.append((sentence1, similar_sentence, max_similarity))
    return similar_sentences

# Function to highlight differences between two sentences
def highlight_differences(sentence1, sentence2):
    diff = ""
    for i, s in enumerate(difflib.ndiff(sentence1, sentence2)):
        if s[0] == ' ':
            diff += s[2]
        elif s[0] == '-':
            diff += f"<span style='background-color: #BFBFBF;'>{s[2]}</span>"
        elif s[0] == '+':
            diff += f"<span style='background-color: #BFBFBF;'>{s[2]}</span>"
    return diff

# Function to process the documents
def process_documents(asr_dataset_path, qa_dataset_path):
    delimiter = "."

        # Process ASR dataset
    with open(asr_dataset_path, 'r') as file:
        bytes_data_asr = file.read()
        list_text_asr = [x + delimiter for x in bytes_data_asr.lower().split(delimiter) if x]
        st.write("ASR Documents:")
        for i, text in enumerate(list_text_asr):
            st.write(f"d{i+1}: {text}")

    # Process QA dataset
    with open(qa_dataset_path, 'r') as file:
        bytes_data_qa = file.read()
        list_text_qa = [x + delimiter for x in bytes_data_qa.lower().split(delimiter) if x]
        st.write("QA Documents:")
        for i, text in enumerate(list_text_qa):
            st.write(f"d{len(list_text_asr)+i+1}: {text}")

    # Tokenization and Stemming
    tokenized_asr = [tokenize_and_stem(text) for text in list_text_asr]
    tokenized_qa = [tokenize_and_stem(text) for text in list_text_qa]

    # Create CountVectorizer and transform datasets
    max_features = max(len(list_text_asr), len(list_text_qa))

    # Vectorize and transform ASR dataset
    LemDocuments_asr = CountVectorizer(tokenizer=tokenize_and_stem, max_features=max_features)
    tf_matrix_asr = LemDocuments_asr.fit_transform(list_text_asr)

    # Vectorize and transform QA dataset
    LemDocuments_qa = CountVectorizer(tokenizer=tokenize_and_stem, max_features=max_features)
    tf_matrix_qa = LemDocuments_qa.fit_transform(list_text_qa)

    # Display Vocabulary
    st.write("Vocabulary ASR:")
    vocabulary_asr_html = '<span style="color:black">' + str(LemDocuments_asr.vocabulary_) + '</span>'
    st.markdown(vocabulary_asr_html, unsafe_allow_html=True)

    st.write("Vocabulary QA:")
    vocabulary_qa_html = '<span style="color:black">' + str(LemDocuments_qa.vocabulary_) + '</span>'
    st.markdown(vocabulary_qa_html, unsafe_allow_html=True)

    # Perform stemming on datasets
    stemmed_asr_dataset = [stem_sentence(sentence) for sentence in list_text_asr]
    st.write("Stemmed ASR dataset:")
    st.write(stemmed_asr_dataset)

    stemmed_qa_dataset = [stem_sentence(sentence) for sentence in list_text_qa]
    st.write("Stemmed QA dataset:")
    st.write(stemmed_qa_dataset)

    # Create TF-IDF transformer and transform TF matrix
    tfidfTran_asr = TfidfTransformer(norm="l2")
    tfidfTran_asr.fit(tf_matrix_asr)
    tfidf_matrix_asr = tfidfTran_asr.transform(tf_matrix_asr)
    st.write("TF-IDF matrix (ASR dataset):")
    st.write(tfidf_matrix_asr.toarray())

    tfidfTran_qa = TfidfTransformer(norm="l2")
    tfidfTran_qa.fit(tf_matrix_qa)
    tfidf_matrix_qa = tfidfTran_qa.transform(tf_matrix_qa)
    st.write("TF-IDF matrix (QA dataset):")
    st.write(tfidf_matrix_qa.toarray())

    # Calculate cosine similarity matrices
    padded_tfidf_matrix_asr = pad_matrix(tfidf_matrix_asr.toarray(), (70, 70))
    padded_tfidf_matrix_qa = pad_matrix(tfidf_matrix_qa.toarray(), (70, 70))

    if padded_tfidf_matrix_asr.shape[1] == padded_tfidf_matrix_qa.shape[0]:
        cos_similarity_matrix = cosine_similarity(padded_tfidf_matrix_asr, padded_tfidf_matrix_qa)
        st.write("Similarity matrix (ASR vs QA):")
        st.write(pd.DataFrame(cos_similarity_matrix))

        # Calculate kappa index for Similarity matrices
        kappa_score, index = kappa_index(cos_similarity_matrix[0], cos_similarity_matrix[1])
        st.write("Kappa Index (ASR vs QA):", kappa_score, index)
    else:
        st.write("Error: Dimension mismatch between ASR and QA datasets.")

# Load Word2Vec model
@st.cache_resource()
def load_word2vec_model():
    return api.load('word2vec-google-news-300')

# Function for text preprocessing
def preprocess_documents(documents):
    preprocessed_documents = []
    for doc in documents:
        tokens = word_tokenize(doc.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_documents.append(tokens)
    return preprocessed_documents

# Function to compute semantic similarity matrix
def compute_semantic_similarity_matrix(model, docs1, docs2):
    similarity_matrix = np.zeros((len(docs1), len(docs2)))

    for i, doc1 in enumerate(docs1):
        for j, doc2 in enumerate(docs2):
            vec1 = np.mean([model[word] for word in doc1 if word in model], axis=0)
            vec2 = np.mean([model[word] for word in doc2 if word in model], axis=0)

            if vec1.size == 0 or vec2.size == 0:
                continue

            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarity_matrix[i, j] = similarity

    return similarity_matrix

# Function to find best mappings and recommendations
def find_best_mappings(similarity_matrix, asr_lines, qa_lines):
    best_mappings = []
    for i, sim_row in enumerate(similarity_matrix):
        max_sim_index = np.argmax(sim_row)
        max_sim_value = sim_row[max_sim_index]

        if max_sim_value >= 0.8:
            recommendation = "High Semantic Similarity"
        elif max_sim_value >= 0.7:
            recommendation = "Moderate Semantic Similarity"
        else:
            recommendation = "Recommended for improvement"

        best_mappings.append({
            "ASR Document": asr_lines[i],
            "QA Document": qa_lines[max_sim_index],
            "Semantic Similarity Score": max_sim_value,
            "Recommendation": recommendation
        })

    return best_mappings

def main():
    st.title('Mapping ASRs and QAs Documents')

    st.markdown("""
            This application maps ASRs (Architecturally Significant Requirements) documents to QAs (Quality Attributes) documents
            based on Semantic Similarity using Word2Vec model.
            """)

    # File upload
    asr_dataset = st.file_uploader("Upload ASR dataset", type="txt")
    qa_dataset = st.file_uploader("Upload QA dataset", type="txt")

    if asr_dataset is not None and qa_dataset is not None:
        # Save uploaded files to temporary locations
        asr_dataset_path = os.path.join(os.getcwd(), "asr_dataset.txt")
        qa_dataset_path = os.path.join(os.getcwd(), "qa_dataset.txt")

        with open(asr_dataset_path, "w") as asr_file, open(qa_dataset_path, "w") as qa_file:
            asr_file.write(asr_dataset.getvalue().decode("utf-8"))
            qa_file.write(qa_dataset.getvalue().decode("utf-8"))

        # Add a 'Process' button to trigger document processing
        if st.button('Process'):
            process_documents(asr_dataset_path, qa_dataset_path)

            # Load and preprocess documents for semantic similarity
            with open(asr_dataset_path, "r") as asr_file:
                asr_lines = asr_file.readlines()

            with open(qa_dataset_path, "r") as qa_file:
                qa_lines = qa_file.readlines()

            asr_documents = preprocess_documents(asr_lines)
            qa_documents = preprocess_documents(qa_lines)

            word2vec_model = load_word2vec_model()
            similarity_matrix = compute_semantic_similarity_matrix(word2vec_model, asr_documents, qa_documents)

            # Find best mappings and recommendations
            best_mappings = find_best_mappings(similarity_matrix, asr_lines, qa_lines)

            # Display best mappings and recommendations in a table
            st.subheader("Best Mapping ASR Documents to QA Documents")
            best_mappings_df = pd.DataFrame(best_mappings)
            st.table(best_mappings_df)

            # Additional option to display similarity matrix (formatted as table)
            st.subheader("Semantic Similarity Matrix")

            # Create a DataFrame with similarity scores formatted as text
            table_data = []
            for i in range(len(asr_documents)):
                table_row = []
                for j in range(len(qa_documents)):
                    similarity_score = similarity_matrix[i, j]
                    table_row.append(f"{similarity_score:.2f}")
                table_data.append(table_row)

            # Create a DataFrame for display
            df = pd.DataFrame(table_data, columns=[f"QA {j+1}" for j in range(len(qa_documents))],
                              index=[f"ASR {i+1}" for i in range(len(asr_documents))])

            st.write(df)

if __name__ == "__main__":
    main()
