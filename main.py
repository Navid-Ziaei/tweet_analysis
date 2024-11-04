import random
import torch
import numpy as np
from transformers import AutoTokenizer

from src import analyze_across_stanford, predict_label_stanford
from src.feature_extraction import TextFeatureExtractor
from src.model.spicy_model.model import analyze_across_texts_spacy, predict_label_spacy
from src.model.supervised_models.transformer_base_models import fine_tune_and_evaluate, fine_tune_and_evaluate2, \
    load_model_and_predict
from src.model.supervised_models.xgboost_test_classifier import XGBoostTextClassifier
from src.model.transformer_base_tagging.model_pos_tag import analyze_across_texts_transformers, predict_label_bert
from src.model.utils.train_utils import evaluate_ner_models
from src.settings import Settings, Paths
from src.data_handler import DatasetHandler
import nltk

nltk.download('punkt')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def main():
    # Load settings from settings.json
    settings = Settings()
    settings.load_settings()

    # Set up paths for data
    paths = Paths(settings)
    paths.load_device_paths()
    paths.create_paths()

    # load and preprocess data
    data_handler = DatasetHandler(paths)
    data_handler.load_data()
    data_handler.preprocess_data()
    # data_handler.explanatory_data_analysis()

    data = data_handler.data['clean_text'].str.lower().values
    labels = data_handler.data['is_related_to_Across_protocol']

    model = 'xgboost'

    # training
    if model == 'xgboost':
        text_classifier = XGBoostTextClassifier(data, labels, mode='weight')
        text_classifier.train_model(save_path=paths.path_result)
    elif model == 'unsupervised':
        pass
    elif model == 'transformer':
        results = fine_tune_and_evaluate2('bert-base-uncased', data, labels, paths)
    else:
        raise ValueError("Model is not defined")


    if model == 'xgboost':
        text_classifier = XGBoostTextClassifier(data, labels, mode='weight')
        output = text_classifier.load_and_predict(paths.model_path,texts)
    elif model == 'unsupervised':
        output = predict_label_spacy(texts, model_name="en_core_web_trf")
    elif model == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        output = load_model_and_predict(paths.model_path, tokenizer, texts)




    # text = ("across redefines")
    # text = text.lower()
    # print(predict_label_stanford(text))
    # print(predict_label_spacy(text, model_name="en_core_web_md"))
    # print(predict_label_bert(text, model_name="dslim/bert-base-NER-uncased"))

    # evaluate_ner_models(data, data_handler, labels, paths)







if __name__ == "__main__":
    main()
