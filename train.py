import random
import torch
import numpy as np
import nltk
from src.model.supervised_models.transformer_base_models import fine_tune_and_evaluate
from src.model.supervised_models.xgboost_test_classifier import XGBoostTextClassifier
from src.model.utils.train_utils import evaluate_ner_models
from src.settings import Settings, Paths
from src.data_handler import DatasetHandler


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

    for model in settings.method_list:
        # training
        if model.lower() == 'xgboost':
            text_classifier = XGBoostTextClassifier(data, labels, mode='weight')
            text_classifier.train_model(save_path=paths.path_result)
        elif model.lower() == 'unsupervised':
            evaluate_ner_models(data, data_handler, labels, paths)
        elif model.lower() == 'transformer':
            results = fine_tune_and_evaluate('bert-base-uncased', data, labels, paths, sampling_method='undersample')
        else:
            raise ValueError("Model is not defined")

if __name__ == '__main__':
    main()
