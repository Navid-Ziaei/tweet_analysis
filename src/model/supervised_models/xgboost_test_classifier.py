import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, f1_score, \
    recall_score, precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_sample_weight
import spacy
#from xgboost.testing.data import joblib


class XGBoostTextClassifier:
    def __init__(self, data, labels, mode='none'):
        self.data = data
        self.labels = labels
        self.mode = mode
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.nlp = spacy.load('en_core_web_sm')
        self.results = []


    def preprocess_documents(self):
        preprocessed_docs = [
            ' '.join(token.lemma_.lower() for token in self.nlp(doc) if token.is_alpha and not token.is_stop) for doc in
            self.data]
        return preprocessed_docs

    def vectorize_text(self):
        preprocessed_docs = self.preprocess_documents()
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_docs)

        return tfidf_matrix

    def train_model(self, save_path):
        X = self.vectorize_text()
        y = np.array(self.labels)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_model = None
        best_score = 0

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.mode == 'undersample':
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                self.model.fit(X_train, y_train)
            elif self.mode == 'oversample':
                ros = RandomOverSampler(random_state=42)
                X_train, y_train = ros.fit_resample(X_train, y_train)
                self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                self.model.fit(X_train, y_train)
            elif self.mode == 'weight':
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
                self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                self.model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            self.results.append({
                'fold': fold+1,
                'accuracy': accuracy,
                'precision': precision_score(y_test, y_pred, pos_label=1),
                'recall': recall_score(y_test, y_pred, pos_label=1),
                'f1_score': f1_score(y_test, y_pred, pos_label=1)
            })
            if f1 > best_score:
                best_score = f1
                best_model = self.model
                pickle.dump(best_model, open(os.path.join(save_path, f"best_xgb_model.pkl"), "wb"))
                with open(os.path.join(save_path, f'vectorizer.pkl'), 'wb') as fin:
                    pickle.dump(self.vectorizer, fin)

            y_pred = self.model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title('Confusion Matrix of Best Model')
            plt.tight_layout()
            plt.savefig(save_path + f"XGboost_confusion_matrix_fold{fold}.png")
            plt.show()

            best_features = self.get_important_features()
            best_features = pd.DataFrame(best_features)
            best_features.to_csv(save_path + f"XGboost_important_features_fold{fold}.csv")

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(save_path + "fold_results_xgboost.csv", index=False)
        print(results_df)
        mean_results = results_df.mean()
        std_results = results_df.std()
        summary_df = pd.DataFrame({'mean': mean_results, 'std': std_results})
        summary_df.to_csv(save_path + "results_summary_xgboost.csv", index=True)

        # Plot confusion matrix for the best model

    def load_and_predict(self, model_path, new_data):
        model = pickle.load(open(os.path.join(model_path, f"best_xgb_model.pkl"), "rb"))
        with open(os.path.join(model_path, f'vectorizer.pkl'), 'rb') as file:
            loaded_vectorizer_pickle = pickle.load(file)

        new_preprocessed_docs = [
            ' '.join(token.lemma_.lower() for token in self.nlp(doc) if token.is_alpha and not token.is_stop) for doc in
            new_data]
        new_tfidf_matrix = loaded_vectorizer_pickle.transform(new_preprocessed_docs)
        predictions = model.predict(new_tfidf_matrix)

        output = []
        for text, pred_label in zip(new_data, predictions):
            output.append(
                {'text': text,
                 'label': int(pred_label)}
            )
        return output, predictions
    def get_important_features(self):
        feature_importances = self.model.feature_importances_
        feature_names = self.vectorizer.get_feature_names_out()
        feature_importances_with_names = sorted(zip(feature_importances, feature_names), reverse=True)
        return feature_importances_with_names[:20]  # Adjust number as needed for more features



