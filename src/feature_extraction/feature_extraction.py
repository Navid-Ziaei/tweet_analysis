import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

class TextFeatureExtractor:
    def __init__(self, nlp_model='en_core_web_sm'):
        self.nlp = spacy.load(nlp_model)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def tokenize(self, text):
        return [token.text for token in self.nlp(text)]

    def extract_pos_tags(self, text):
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def extract_dependency_parse(self, text):
        doc = self.nlp(text)
        return [(token.text, token.dep_, token.head.text) for token in doc]

    def extract_word_embeddings(self, text):
        doc = self.nlp(text)
        return {token.text: token.vector for token in doc if not token.is_stop and token.is_alpha}

    def extract_tfidf_features(self, corpus):
        return self.tfidf_vectorizer.fit_transform(corpus)

    def extract_ner(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_keyword_co_occurrence(self, text, keywords):
        doc = self.nlp(text.lower())
        occurrences = {keyword: False for keyword in keywords}
        for token in doc:
            if token.text in occurrences:
                occurrences[token.text] = True
        return occurrences

    def context_features(self, text, target='across'):
        doc = self.nlp(text.lower())
        features = {}
        for token in doc:
            if token.text == target:
                left = token.i - 1 if token.i > 0 else None
                right = token.i + 1 if token.i < len(doc) - 1 else None
                features['prev_word'] = doc[left].text if left is not None else ''
                features['next_word'] = doc[right].text if right is not None else ''
                features['is_start'] = token.i == 0
                features['is_end'] = token.i == len(doc) - 1
        return features

    def extract_features(self, text):
        features = {
            'pos_tags': self.extract_pos_tags(text),
            'dependencies': self.extract_dependency_parse(text),
            'word_embeddings': self.extract_word_embeddings(text),
            'named_entities': self.extract_ner(text),
            'context_features': self.context_features(text)
        }
        return features


