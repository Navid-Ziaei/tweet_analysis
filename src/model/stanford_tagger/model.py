from nltk.tag.stanford import StanfordNERTagger
import os

def stanford_tag_ner(document):
    """
    Load the Stanford NER Tagger and tag the given document.

    Args:
    model_path (str): Path to the Stanford NER model.
    jar_path (str): Path to the Stanford NER jar file.
    document (str): Text document to tag.

    Returns:
    list of tuples: List of word and its corresponding NER tag.
    """
    working_folder = os.path.abspath(__file__)
    working_folder = "\\".join(working_folder.split("\\")[:-1]) + "\\"
    jar = working_folder + "stanford-ner-2015-04-20/stanford-ner-3.5.2.jar"
    model = working_folder + "stanford-ner-2015-04-20/classifiers/"
    st = StanfordNERTagger(model + "english.conll.4class.distsim.crf.ser.gz", jar, encoding='utf8')

    # Tokenize the document
    tokens = document.split()

    # Tag the document using the Stanford NER tagger
    ner_tags = st.tag(tokens)
    is_org = False
    for word, tag in ner_tags:
        if word == 'across':
            if tag != 'O':
                is_org = True


    return is_org