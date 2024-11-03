import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.settings import Settings, Paths
from src.data_handler import DatasetHandler
from src.model.stanford_tagger import stanford_tag_ner
import pandas as pd
import nltk
import spacy

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

    data = data_handler.data['clean_text'].values
    labels = data_handler.data['is_related_to_Across_protocol']
    predicted_labels = []
    for text in tqdm(data):
        predicted_label = stanford_tag_ner(text)
        predicted_labels.append(predicted_label)

    # Load the pre-trained English model
    nlp = spacy.load("en_core_web_trf")

    # Process the sentence
    for text, label in zip(data, labels):
        doc = nlp(text)

        # Print POS tags
        for token in doc:
            if token.text == 'across':
                print(f'{token.text}: {token.pos_}')
        print(label)



    predictions = []
    for text, label in zip(data, labels):
        # Process the text through the NLP pipeline
        doc = nlp(text)

        # Initialize variable to track if 'across' related to Across Protocol is found
        found_across_protocol = 0

        # Search for 'across' and check surrounding entity tags
        for ent in doc.ents:
            if "across" in ent.text.lower():  # Check if 'across' is part of an entity
                found_across_protocol = 1
                print(text)
                print(f"label is {label}")
                break

        # Append prediction for this text
        predictions.append(found_across_protocol)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print("Accuracy of 'across' related to Across Protocol detection:", accuracy)



if __name__ == "__main__":
    main()
