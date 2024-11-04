from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def predict_label_bert(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pos_pipeline = pipeline("token-classification", model=model_name)

    # Process each text and analyze the word 'across'
    chunks = process_text(text, tokenizer)
    predicted_label = 0

    for chunk in chunks:
        pos_tags = pos_pipeline(chunk)
        for tag in pos_tags:
            word = tag['word'].replace('##', '')  # Fix tokenization issues common with BERT
            if word.lower() == 'across':
                print(f"{word}: {tag['entity']} ")
                pos_tag = tag['entity']  # Collect POS tags from the transformers output
                # ent_type = tag['entity_group']  # Collect entity type
                if pos_tag != 'ADP':  # Assume 'ADP' stands for adpositions/prepositions
                    predicted_label = 1
                    break
        if predicted_label == 1:
            break  # Stop processing further chunks if 'across' is already processed as significant

    return predicted_label


def process_text(text, tokenizer, max_len=10):  # 510 + 2 for special tokens [CLS] and [SEP]
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_len:
        return [text]  # Return the original text if it's short enough

    # Find all occurrences of 'across' and center them in the chunks
    positions = [i for i, token in enumerate(tokens) if token == 'across']
    chunks = []
    for pos in positions:
        start = max(0, pos)
        end = start + max_len
        if end > len(tokens):
            end = len(tokens)
            start = max(0, end - max_len)
        chunk = tokens[start:end]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks


def analyze_across_texts_transformers(model_name, data, labels):
    """
    Analyzes texts to determine the role of the word 'across' using a BERT-based POS tagging model and calculate the prediction accuracy.

    Args:
    model_name (str): The transformer model to load for POS tagging.
    data (list of str): The texts to analyze.
    labels (list of int): The ground truth labels for each text indicating if 'across' is relevant.

    Returns:
    tuple: A tuple containing the predicted labels, list of POS tags, list of entity types.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pos_pipeline = pipeline("token-classification", model=model_name)

    predicted_labels = []
    pos_list = []

    # Process each text and analyze the word 'across'
    for text in tqdm(data):
        chunks = process_text(text, tokenizer)
        predicted_label = 0
        pos_tag = ''
        ent_type = ''

        for chunk in chunks:
            pos_tags = pos_pipeline(chunk)
            for tag in pos_tags:
                word = tag['word'].replace('##', '')  # Fix tokenization issues common with BERT
                if word.lower() == 'across':
                    print(f"{word}: {tag['entity']} ")
                    pos_tag = tag['entity']  # Collect POS tags from the transformers output
                    #ent_type = tag['entity_group']  # Collect entity type
                    if pos_tag != 'ADP':  # Assume 'ADP' stands for adpositions/prepositions
                        predicted_label = 1
                        break
            if predicted_label == 1:
                break  # Stop processing further chunks if 'across' is already processed as significant

        predicted_labels.append(predicted_label)
        pos_list.append(pos_tag)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(labels, predicted_labels)
    print("Accuracy of 'across' related to Across Protocol detection:", accuracy)

    return predicted_labels, pos_list
