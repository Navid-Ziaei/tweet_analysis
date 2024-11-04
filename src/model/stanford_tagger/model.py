from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
import os

from sklearn.metrics import accuracy_score
from tqdm import tqdm


def predict_label_stanford(text):
    working_folder = os.path.abspath(__file__)
    working_folder = "\\".join(working_folder.split("\\")[:-1]) + "\\"

    model_ner = working_folder + "stanford-ner-2015-04-20/classifiers/english.conll.4class.distsim.crf.ser.gz"
    jar_ner = working_folder + "stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"

    model_tagger = working_folder + 'stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
    jar_tagger = working_folder + 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'

    # Initialize Stanford NER and POS taggers
    ner_tagger = StanfordNERTagger(model_ner, jar_ner, encoding='utf8')
    pos_tagger = StanfordPOSTagger(model_tagger, jar_tagger, encoding='utf8')

    tokens = text.split()
    predicted_label = 0

    try:
        ner_tags = ner_tagger.tag(tokens)
        pos_tags = pos_tagger.tag(tokens)

        for (word, ner), (_, pos) in zip(ner_tags, pos_tags):
            if word.lower() == 'across':
                if ner != 'O' or pos != 'IN':
                    predicted_label = 1
                break  # Break after 'across' to avoid repeating if multiple 'across' present
    except:
        print(text)
    return predicted_label


def analyze_across_stanford(data, labels):
    """
    Analyzes texts to determine the role of the word 'across' using Stanford NER and POS taggers
    to calculate the prediction accuracy and returns various tagging details.

    Args:
    model_path (str): Path to the Stanford model directory.
    jar_path (str): Path to the Stanford NER and POS jar files.
    data (list of str): The texts to analyze.
    labels (list of int): Ground truth labels indicating if 'across' is relevant as a named entity.

    Returns:
    tuple: A tuple containing the predicted labels, list of POS tags, list of entity types.
    """
    working_folder = os.path.abspath(__file__)
    working_folder = "\\".join(working_folder.split("\\")[:-1]) + "\\"

    model_ner = working_folder + "stanford-ner-2015-04-20/classifiers/english.conll.4class.distsim.crf.ser.gz"
    jar_ner = working_folder + "stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"

    model_tagger = working_folder + 'stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger'
    jar_tagger = working_folder + 'stanford-postagger-full-2020-11-17/stanford-postagger.jar'

    # Initialize Stanford NER and POS taggers
    ner_tagger = StanfordNERTagger(model_ner, jar_ner, encoding='utf8')
    pos_tagger = StanfordPOSTagger(model_tagger, jar_tagger, encoding='utf8')

    predicted_labels = []
    pos_list = []
    ent_type_list = []

    # Process each text and analyze the word 'across'
    for text in tqdm(data):
        tokens = text.split()


        predicted_label = 0
        pos_tag = ''
        ent_type = ''

        try:
            ner_tags = ner_tagger.tag(tokens)
            pos_tags = pos_tagger.tag(tokens)

            for (word, ner), (_, pos) in zip(ner_tags, pos_tags):
                if word.lower() == 'across':
                    if ner != 'O' or pos != 'IN':
                        predicted_label = 1
                    pos_tag = pos
                    ent_type = ner
                    break  # Break after 'across' to avoid repeating if multiple 'across' present
        except:
            print(text)
        predicted_labels.append(predicted_label)
        pos_list.append(pos_tag)
        ent_type_list.append(ent_type)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(labels, predicted_labels)
    print("Accuracy of 'across' related to Across Protocol detection:", accuracy)

    return predicted_labels, pos_list, ent_type_list
