from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class TextAnalysis:
    def __init__(self, documents):
        self.nlp = spacy.load('en_core_web_sm')
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def preprocess_documents(self):
        # Tokenize, remove stopwords, non-alphabetic characters, and lowercase
        preprocessed_docs = [' '.join(token.lemma_.lower() for token in self.nlp(doc) if token.is_alpha and not token.is_stop) for doc in self.documents]
        return preprocessed_docs

    def vectorize_text(self):
        # Convert text data into TF-IDF matrix
        preprocessed_docs = self.preprocess_documents()
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_docs)
        return tfidf_matrix

    def apply_pca(self, n_components=2):
        # Reduce dimensions using PCA
        tfidf_matrix = self.vectorize_text()
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(tfidf_matrix.toarray())
        return reduced_data

    def plot_pca_results(self):
        # Visualize the PCA-transformed data
        reduced_data = self.apply_pca()
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
        plt.title('PCA Results of Text Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

# Example usage
class TextCluster:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.model = None
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Cleans the input text by removing non-alphabetic characters, stopwords.

        Args:
        text (str): Input text to be cleaned.

        Returns:
        str: Cleaned text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
        tokens = word_tokenize(text.lower())  # Tokenize and convert to lower case
        cleaned_text = ' '.join([word for word in tokens if word not in self.stop_words and len(word) > 1])
        return cleaned_text

    def preprocess(self):
        # Clean data and convert text data into TF-IDF features
        cleaned_data = [self.clean_text(text) for text in self.data]
        X = self.vectorizer.fit_transform(cleaned_data)
        return X

    def apply_clustering(self, num_clusters=2):
        # Applying KMeans clustering
        self.model = KMeans(n_clusters=num_clusters, random_state=42)
        labels = self.model.fit_predict(self.preprocess())
        return labels

    def visualize_clusters(self, labels):
        # Use PCA to reduce dimensions for visualization
        X = self.preprocess()
        reduced_data = PCA(n_components=2).fit_transform(X.toarray())

        # Plotting the clusters
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50,
                    alpha=0.6)
        plt.title('Clusters of Text Data')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar()
        plt.show()



    def infer_across_usage(self, labels):
        # Analyze cluster results to infer 'across' usage
        # This function assumes one of the clusters pertains to the specific use of "Across Protocol"
        for i, text in enumerate(self.data):
            if 'across' in text:
                print(f"Text: {text}\nCluster: {labels[i]}\n")
