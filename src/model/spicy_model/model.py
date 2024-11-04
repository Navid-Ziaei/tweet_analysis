import spacy
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def predict_label_spacy(text, model_name):
    nlp = spacy.load(model_name)
    doc = nlp(text)
    predicted_label = 0
    for token in doc:
        if token.text.lower() == 'across':
            # If 'across' has a POS that is not 'ADP' (adposition) or has an entity type
            if token.pos_ != 'ADP' or token.ent_type_ != '':
                predicted_label = 1
                break

    return predicted_label
def analyze_across_texts_spacy(model_name, data, labels):
    """
    Analyzes texts to determine the role of the word 'across' and calculate the prediction accuracy.

    Args:
    model_name (str): The spaCy model to load.
    data (list of str): The texts to analyze.
    labels (list of int): The ground truth labels for each text.

    Returns:
    tuple: A tuple containing the predicted labels, list of POS tags, list of entity types.
    """
    nlp = spacy.load(model_name)

    predicted_labels = []
    pos_list = []
    ent_type_list = []

    # Process each text and analyze the word 'across'
    for text in tqdm(data):
        doc = nlp(text)
        predicted_label = 0
        pos_tag = ''
        ent_type = ''

        for token in doc:

            if token.text.lower() == 'across':
                # If 'across' has a POS that is not 'ADP' (adposition) or has an entity type
                if token.pos_ != 'ADP' or token.ent_type_ != '':
                    predicted_label = 1
                    pos_tag = token.pos_  # Collect POS tags
                    ent_type = token.ent_type_  # Collect
                    break
                else:
                    pos_tag = token.pos_  # Collect POS tags
                    ent_type = token.ent_type_  # Collect

        predicted_labels.append(predicted_label)
        pos_list.append(pos_tag)
        ent_type_list.append(ent_type)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(labels, predicted_labels)
    print("Accuracy of 'across' related to Across Protocol detection:", accuracy)

    return predicted_labels, pos_list, ent_type_list
